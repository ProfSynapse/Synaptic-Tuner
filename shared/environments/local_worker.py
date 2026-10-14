"""Fixed local fixture child supervisor, implementing EnvironmentRuntime.

No shell, user code, provider client or credentials cross this boundary. Process
custody begins at acknowledged fixed-guardian handoff, not restorable from a PID
or serialized data. Owner death triggers guardian termination and cleanup; guardian
death, pre-handoff death and machine loss remain outside that guarantee.
This is process supervision and bounded fixture I/O, not a security sandbox.
Popen construction and parent filesystem setup/cleanup are synchronous OS calls;
finite IPC/termination waits do not impose a general wall-clock bound on them.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import select
import struct
import subprocess
import sys
import time

from .base import EnvironmentRuntime
from .bounded_local_runtime import BoundedLocalEnvironmentRuntime, FilesystemLimits
from .fixture_parser import EnvironmentFixture


_OPERATIONS = {'mkdir': 1, 'write_text': 2, 'read_text': 1, 'list_dir': 1,
               'move': 3, 'copy': 3, 'delete': 2, 'exists': 1, 'search': 2, 'snapshot': 1}
_ERRORS = {'FileNotFoundError': FileNotFoundError, 'FileExistsError': FileExistsError,
           'IsADirectoryError': IsADirectoryError, 'ValueError': ValueError,
           'TypeError': TypeError, 'TimeoutError': TimeoutError, 'OSError': OSError,
           'RuntimeError': RuntimeError, 'UnicodeDecodeError': ValueError}
_MAX_FRAME = 128 * 1024 * 1024



class WorkerLifecycleError(RuntimeError):
    """Closed supervisor failure; never a candidate assertion/model-quality error."""
    def __init__(self, code):
        if code not in {'worker_deadline_exceeded', 'worker_boundary_interrupted',
                        'worker_protocol_invalid', 'worker_start_uncertain',
                        'worker_termination_uncertain', 'worker_cleanup_uncertain',
                        'worker_unavailable'}:
            raise ValueError('invalid worker lifecycle code')
        self.code = code
        super().__init__(code)


def _encode(value, maximum):
    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(raw) > maximum:
        raise ValueError('worker_transport_limit')
    return struct.pack('!I', len(raw)) + raw


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('worker_protocol_invalid')
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError('worker_protocol_invalid')))
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError('worker_protocol_invalid') from None


class WorkerEnvironmentRuntime(EnvironmentRuntime):
    """Effect-free factory product; setup consumes this instance exactly once.

    ``boundary`` is a parent-only callable, invoked on the calling/owning thread;
    it may raise cancellation or host deadline errors. ``deadline`` optionally
    clamps the original lifetime and is never renewed by another operation.
    IPC faults/lost replies invalidate the worker, terminate/wait and clean only
    after confirmed exit. Cleanup failure retains descriptor custody for retrying
    cleanup, never execution. Copy/move retain bounded runtime semantics.
    """
    def __init__(self, private_root, limits, *, boundary=None, deadline=None):
        if type(limits) is not FilesystemLimits:
            raise TypeError('exact FilesystemLimits required')
        if boundary is not None and not callable(boundary):
            raise TypeError('boundary must be callable')
        if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
            raise ValueError('deadline must be finite')
        self._storage = BoundedLocalEnvironmentRuntime(private_root, limits)
        self.limits = limits
        self._boundary = boundary
        self._original_deadline = deadline
        self._deadline = None
        self._process = None
        self._used = False
        self._failed = False
        self._spawn_started = False
        self._spawn_uncertain = False
        self._request_id = 0
        self._owner_write = None
        self._status_read = None
        self._status_buffer = bytearray()
        self._worker_pid = None
        self._shutdown_status = None
        self._custody = False
        self._initializing = False
        self._input_bound = min(_MAX_FRAME, 6 * limits.max_content_bytes +
                                limits.max_entries * (6 * limits.max_path_bytes + 64) + 8192)
        self._output_bound = min(_MAX_FRAME, 6 * max(limits.max_output_bytes, limits.max_snapshot_bytes) + 8192)

    def _guard(self):
        if self._boundary is not None:
            try:
                self._boundary()
            except Exception:
                raise WorkerLifecycleError('worker_boundary_interrupted') from None
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise WorkerLifecycleError('worker_deadline_exceeded')

    def _prevalidate(self, fixture):
        if type(fixture) is not EnvironmentFixture or type(fixture.files) is not dict or type(fixture.directories) is not list:
            raise TypeError('exact fixture required')
        if len(fixture.files) + len(fixture.directories) > self.limits.max_entries:
            raise ValueError('fixture_entry_limit')
        directories = set()
        total = 0
        for path in fixture.directories:
            parts = self._storage._parts(path, root=True)
            directories.update(self._storage._parents(parts))
            if parts:
                directories.add(path)
            if len(directories) > self.limits.max_entries:
                raise ValueError('fixture_entry_limit')
        for path, text in fixture.files.items():
            parts = self._storage._parts(path)
            if type(text) is not str or len(text) > self.limits.max_content_bytes:
                raise ValueError('fixture_content_limit')
            total += len(text.encode('utf-8'))
            directories.update(self._storage._parents(parts))
            if total > min(self.limits.max_content_bytes, (_MAX_FRAME - 8192) // 6) or len(directories) + len(fixture.files) > self.limits.max_entries:
                raise ValueError('fixture_content_limit')
        if directories.intersection(fixture.files):
            raise ValueError('fixture_path_conflict')
        # Ensure transport fits before scratch effects, too.
        _encode({'directories': fixture.directories, 'files': fixture.files}, self._input_bound)

    def setup(self, fixture):
        if self._used:
            raise RuntimeError('worker_instance_consumed')
        self._prevalidate(fixture)
        self._guard()
        self._used = True
        self._deadline = min(time.monotonic() + self.limits.max_seconds,
                             self._original_deadline if self._original_deadline is not None else math.inf)
        self._guard()
        try:
            self._storage.setup(EnvironmentFixture())
            self._storage._deadline = self._deadline
            self._guard()
            descriptors = tuple({self._storage._root_fd, *(fd for _, _, fd in self._storage._anchors)})
            entry = Path(__file__).with_name('local_worker_child.py').resolve(strict=True)
            allocated = []
            try:
                for _ in range(4):
                    allocated.append(os.pipe())
            except BaseException:
                for pair in allocated:
                    for fd in pair:
                        os.close(fd)
                raise
            (command_read, command_write), (reply_read, reply_write), \
                (owner_read, owner_write), (status_read, status_write) = allocated
            self._owner_write, self._status_read = owner_write, status_read
            try:
                initial = {'limits': asdict(self.limits), 'anchors': self._storage._anchors,
                    'root_fd': self._storage._root_fd, 'name': self._storage._name,
                    'identity': self._storage._created_identity, 'deadline': float(self._deadline),
                    'command_read': command_read, 'reply_write': reply_write,
                    'owner_read': owner_read, 'status_write': status_write}
                initial_bytes = _encode(initial, 8192)
                self._spawn_started = True
                self._process = subprocess.Popen([sys.executable, '-I', str(entry), 'guardian'],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    env={'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8'},
                    pass_fds=(*descriptors, command_read, reply_write, owner_read, status_write),
                    close_fds=True, shell=False)
                self._initializing = True
                os.set_blocking(self._process.stdin.fileno(), False)
                self._send(initial_bytes)
                self._process.stdin.close()
                self._process.stdin = os.fdopen(command_write, 'wb', buffering=0)
                command_write = None
                self._process.stdout = os.fdopen(reply_read, 'rb', buffering=0)
                reply_read = None
                self._initializing = False
            finally:
                for fd in (command_read, reply_write, owner_read, status_write, command_write, reply_read):
                    if fd is not None:
                        os.close(fd)
            os.set_blocking(self._process.stdin.fileno(), False)
            os.set_blocking(self._process.stdout.fileno(), False)
            self._accept_custody(self._read_status(self._deadline, check_boundary=True))
            self._request('initialize', [{'limits': asdict(self.limits),
                'anchors': self._storage._anchors, 'root_fd': self._storage._root_fd,
                'name': self._storage._name, 'identity': self._storage._created_identity,
                'deadline': float(self._deadline), 'directories': fixture.directories, 'files': fixture.files}])
        except BaseException:
            self._failed = True
            if self._spawn_started and self._process is None:
                self._spawn_uncertain = True
            self.teardown()
            raise

    def _wait_ready(self, fd, *, write=False):
        while True:
            self._guard()
            remaining = self._deadline - time.monotonic()
            ready = select.select([] if write else [fd], [fd] if write else [], [], min(0.05, max(0, remaining)))
            if ready[1] if write else ready[0]:
                return
            if self._process.poll() is not None:
                raise RuntimeError('worker_exited_without_reply')

    def _send(self, raw):
        fd = self._process.stdin.fileno()
        offset = 0
        while offset < len(raw):
            self._wait_ready(fd, write=True)
            try:
                count = os.write(fd, raw[offset:])
            except BlockingIOError:
                continue
            if count <= 0:
                raise RuntimeError('worker_transport_closed')
            offset += count

    def _read_exact(self, count):
        fd = self._process.stdout.fileno()
        raw = bytearray()
        while len(raw) < count:
            self._wait_ready(fd)
            try:
                chunk = os.read(fd, min(65536, count - len(raw)))
            except BlockingIOError:
                continue
            if not chunk:
                raise RuntimeError('worker_transport_closed')
            raw.extend(chunk)
        self._guard()
        return bytes(raw)

    def _reply(self):
        length = struct.unpack('!I', self._read_exact(4))[0]
        if not 0 < length <= self._output_bound:
            raise ValueError('worker_transport_limit')
        return _decode(self._read_exact(length))

    def _validate_args(self, operation, args):
        if operation == 'initialize':
            return
        if operation not in _OPERATIONS or len(args) != _OPERATIONS[operation]:
            raise ValueError('worker_protocol_invalid')
        if operation == 'snapshot':
            if type(args[0]) is not int or args[0] <= 0:
                raise ValueError('invalid_snapshot_limit')
        elif operation == 'search':
            if type(args[0]) is not str or len(args[0]) > self.limits.max_output_bytes:
                raise ValueError('invalid_search_query')
            self._storage._parts(args[1], root=True)
        else:
            self._storage._parts(args[0], root=operation in {'mkdir', 'list_dir', 'exists'})
            if operation == 'write_text':
                if type(args[1]) is not str or len(args[1]) > min(self.limits.max_content_bytes, (_MAX_FRAME - 8192) // 6):
                    raise ValueError('fixture_content_limit')
            elif operation in {'move', 'copy'}:
                self._storage._parts(args[1])
                if type(args[2]) is not bool:
                    raise ValueError('invalid_overwrite_flag')
            elif operation == 'delete' and type(args[1]) is not bool:
                raise ValueError('invalid_recursive_flag')

    def _validate_reply(self, operation, value):
        if operation in {'initialize', 'mkdir', 'write_text', 'copy', 'move', 'delete'}:
            valid = value is None
        elif operation == 'exists':
            valid = type(value) is bool
        elif operation == 'read_text':
            valid = type(value) is str and len(value.encode('utf-8')) <= self.limits.max_output_bytes
        elif operation in {'search', 'list_dir'}:
            valid = type(value) is list and len(value) <= self.limits.max_entries and all(type(item) is str for item in value)
            if valid:
                for item in value:
                    components = self._storage._parts(item)
                    if operation == 'list_dir' and len(components) != 1:
                        raise ValueError('worker_protocol_invalid')
                self._storage._output(value)
        else:
            valid = type(value) is dict and set(value) == {'runtime', 'directories', 'files', 'truncated'} and value['runtime'] == 'bounded_local'
            if valid:
                valid = (type(value['directories']) is list and type(value['files']) is list and
                         type(value['truncated']) is bool and
                         len(value['directories']) + len(value['files']) <= self.limits.max_entries)
            if valid:
                paths = []
                for item in value['directories']:
                    self._storage._parts(item)
                    paths.append(item)
                for item in value['files']:
                    if type(item) is not dict or set(item) != {'path', 'size'} or type(item['size']) is not int or not 0 <= item['size'] <= self.limits.max_content_bytes:
                        raise ValueError('worker_protocol_invalid')
                    self._storage._parts(item['path'])
                    paths.append(item['path'])
                valid = len(paths) == len(set(paths))
                self._storage._output(value, snapshot=True)
        if not valid:
            raise ValueError('worker_protocol_invalid')
        return value

    def _request(self, operation, args):
        if self._failed or self._process is None:
            raise WorkerLifecycleError('worker_unavailable')
        self._validate_args(operation, args)
        self._request_id += 1
        try:
            self._guard()
            self._storage._check()
            self._send(_encode({'id': self._request_id, 'operation': operation, 'args': args}, self._input_bound))
            reply = self._reply()
            self._guard()
            self._storage._check()
            if type(reply) is not dict or set(reply) != {'id', 'ok', 'value'} or reply['id'] != self._request_id or type(reply['id']) is not int or type(reply['ok']) is not bool:
                raise ValueError('worker_protocol_invalid')
            if not reply['ok']:
                if type(reply['value']) is not str or reply['value'] not in _ERRORS:
                    raise ValueError('worker_protocol_invalid')
                if reply['value'] in {'RuntimeError', 'TimeoutError'}:
                    raise WorkerLifecycleError('worker_deadline_exceeded' if reply['value'] == 'TimeoutError' else 'worker_protocol_invalid')
                error = _ERRORS[reply['value']]
            else:
                return self._validate_reply(operation, reply['value'])
        except BaseException as exc:
            self._failed = True
            self.teardown()
            if isinstance(exc, WorkerLifecycleError) or not isinstance(exc, Exception):
                raise
            raise WorkerLifecycleError('worker_protocol_invalid') from None
        # A definite operation error is not an IPC uncertainty or automatic retry.
        raise error('worker_operation_rejected')

    def _accept_custody(self, value):
        if (type(value) is not dict or set(value) != {'custody', 'worker_pid'} or
                value['custody'] is not True or type(value['worker_pid']) is not int or
                value['worker_pid'] <= 0):
            raise WorkerLifecycleError('worker_start_uncertain')
        self._worker_pid = value['worker_pid']
        self._custody = True

    def _read_status(self, deadline, *, check_boundary=False):
        raw = self._status_buffer
        length = struct.unpack('!I', raw[:4])[0] if len(raw) >= 4 else None
        if length is not None and not 0 < length <= 4096:
            raise WorkerLifecycleError('worker_protocol_invalid')
        while length is None or len(raw) < length + 4:
            if check_boundary:
                self._guard()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerLifecycleError('worker_termination_uncertain')
            if not select.select([self._status_read], [], [], min(.05, remaining))[0]:
                continue
            chunk = os.read(self._status_read, (4 if length is None else length + 4) - len(raw))
            if not chunk:
                raise WorkerLifecycleError('worker_termination_uncertain')
            raw.extend(chunk)
            if length is None and len(raw) >= 4:
                length = struct.unpack('!I', raw[:4])[0]
                if not 0 < length <= 4096:
                    raise WorkerLifecycleError('worker_protocol_invalid')
        if len(raw) != length + 4:
            raise WorkerLifecycleError('worker_protocol_invalid')
        value = _decode(raw[4:])
        raw.clear()
        return value

    def teardown(self):
        self._failed = True
        if self._spawn_uncertain:
            raise WorkerLifecycleError('worker_start_uncertain')
        process = self._process
        if process is None:
            for attribute in ('_owner_write', '_status_read'):
                fd = getattr(self, attribute)
                if fd is not None:
                    os.close(fd)
                    setattr(self, attribute, None)
            self._storage.teardown()
            return
        if self._initializing and process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        if self._owner_write is not None:
            os.close(self._owner_write)
            self._owner_write = None
        try:
            if self._shutdown_status is None:
                deadline = time.monotonic() + 3
                value = self._read_status(deadline)
                if type(value) is dict and 'custody' in value:
                    self._accept_custody(value)
                    value = self._read_status(deadline)
                self._shutdown_status = value
            process.wait(timeout=3)
            status = self._shutdown_status
            if (type(status) is not dict or set(status) != {'worker_exited', 'cleanup'} or
                    status['worker_exited'] is not True or type(status['cleanup']) is not bool or
                    process.returncode != 0):
                raise WorkerLifecycleError('worker_termination_uncertain')
        except Exception:
            raise WorkerLifecycleError('worker_termination_uncertain') from None
        for pipe in (process.stdin, process.stdout):
            if pipe is not None and not pipe.closed:
                pipe.close()
        if self._status_read is not None:
            os.close(self._status_read)
            self._status_read = None
        if status['cleanup']:
            try:
                self._storage._close()
            except Exception:
                raise WorkerLifecycleError('worker_cleanup_uncertain') from None
        else:
            # Guardian has authenticated exact worker exit and relinquished cleanup.
            # First report the failed cleanup; a later explicit teardown may retry.
            if not getattr(self, '_cleanup_retry', False):
                self._cleanup_retry = True
                raise WorkerLifecycleError('worker_cleanup_uncertain')
            try:
                self._storage.teardown()
            except Exception:
                raise WorkerLifecycleError('worker_cleanup_uncertain') from None

    def mkdir(self, path):
        return self._request('mkdir', [path])

    def write_text(self, path, content):
        return self._request('write_text', [path, content])

    def read_text(self, path):
        return self._request('read_text', [path])

    def list_dir(self, path='.'):
        return self._request('list_dir', [path])

    def move(self, path, new_path, overwrite=False):
        return self._request('move', [path, new_path, overwrite])

    def copy(self, path, new_path, overwrite=False):
        return self._request('copy', [path, new_path, overwrite])

    def delete(self, path, recursive=False):
        return self._request('delete', [path, recursive])

    def exists(self, path):
        return self._request('exists', [path])

    def search(self, query, path='.'):
        return self._request('search', [query, path])

    def snapshot(self, limit=200):
        return self._request('snapshot', [limit])


def _child_read(maximum):
    header = sys.stdin.buffer.read(4)
    if not header:
        return None
    if len(header) != 4:
        raise ValueError('worker_protocol_invalid')
    size = struct.unpack('!I', header)[0]
    if not 0 < size <= maximum:
        raise ValueError('worker_transport_limit')
    raw = sys.stdin.buffer.read(size)
    if len(raw) != size:
        raise ValueError('worker_protocol_invalid')
    return _decode(raw)


def _run_child():
    runtime = None
    previous = 0
    while True:
        request = _child_read(_MAX_FRAME)
        if request is None:
            return
        if type(request) is not dict or set(request) != {'id', 'operation', 'args'} or type(request['id']) is not int or request['id'] != previous + 1 or type(request['args']) is not list:
            raise ValueError('worker_protocol_invalid')
        previous = request['id']
        operation, args = request['operation'], request['args']
        try:
            if runtime is None:
                if operation != 'initialize' or len(args) != 1:
                    raise ValueError('worker_protocol_invalid')
                initial = args[0]
                limits = FilesystemLimits(**initial['limits'])
                runtime = BoundedLocalEnvironmentRuntime._from_inherited_descriptors(limits,
                    tuple(tuple(row) for row in initial['anchors']), initial['root_fd'], initial['name'],
                    initial['identity'], initial['deadline'])
                for directory in initial['directories']:
                    runtime.mkdir(directory)
                for path, content in initial['files'].items():
                    runtime.write_text(path, content)
                value = None
            else:
                if type(operation) is not str or operation not in _OPERATIONS or len(args) != _OPERATIONS[operation]:
                    raise ValueError('worker_protocol_invalid')
                value = getattr(runtime, operation)(*args)
            reply = {'id': previous, 'ok': True, 'value': value}
        except Exception as exc:
            code = type(exc).__name__ if type(exc).__name__ in _ERRORS else 'RuntimeError'
            reply = {'id': previous, 'ok': False, 'value': code}
        maximum = _MAX_FRAME if runtime is None else min(_MAX_FRAME, 6 * max(runtime.limits.max_output_bytes, runtime.limits.max_snapshot_bytes) + 8192)
        sys.stdout.buffer.write(_encode(reply, maximum))
        sys.stdout.buffer.flush()


def _run_guardian():
    """Own exact child and descriptors; owner EOF is cleanup intent, not recovery."""
    initial = _child_read(8192)
    runtime = BoundedLocalEnvironmentRuntime._from_inherited_descriptors(
        FilesystemLimits(**initial['limits']), tuple(tuple(row) for row in initial['anchors']),
        initial['root_fd'], initial['name'], initial['identity'], initial['deadline'])
    child = None
    status = initial['status_write']
    def publish(value):
        try:
            os.write(status, _encode(value, 4096))
        except BrokenPipeError:
            pass  # Owner died; descriptor cleanup still belongs to guardian.
    try:
        entry = Path(__file__).with_name('local_worker_child.py').resolve(strict=True)
        descriptors = tuple({runtime._root_fd, *(fd for _, _, fd in runtime._anchors)})
        child = subprocess.Popen([sys.executable, '-I', str(entry)],
            stdin=initial['command_read'], stdout=initial['reply_write'], stderr=subprocess.DEVNULL,
            env={'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8'}, pass_fds=descriptors,
            close_fds=True, shell=False)
        os.close(initial['command_read'])
        os.close(initial['reply_write'])
        publish({'custody': True, 'worker_pid': child.pid})
        while child.poll() is None:
            if time.monotonic() >= initial['deadline']:
                break
            if select.select([initial['owner_read']], [], [], 0.05)[0]:
                if os.read(initial['owner_read'], 1) == b'':
                    break
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=2)
        child.wait(timeout=0)
    except Exception:
        # Exit unconfirmed: preserve root, never delete underneath a live worker.
        publish({'worker_exited': False, 'cleanup': False})
        return
    cleanup = False
    try:
        runtime.teardown()
        cleanup = True
    except Exception:
        pass
    publish({'worker_exited': True, 'cleanup': cleanup})
