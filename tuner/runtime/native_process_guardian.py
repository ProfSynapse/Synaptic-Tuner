"""Standalone stdlib guardian for one owned native worker, started with -I -S.

The parent holds a pipe open. EOF (including parent death) terminates the worker
process group. A persisted PID is never reused as restart-time signal authority.
This is POSIX/Darwin process supervision, not a sandbox or Linux pidfd guarantee.
"""

from __future__ import annotations

import errno
import hashlib
import hmac
import json
import os
import resource
import select
import selectors
import signal
import stat
import subprocess
import sys
import time
import types


# Closed exit codes are the only guardian failure detail exposed to the parent.
# They describe where supervision stopped, never an exception or worker output.
_FAILURE_CODES = {
    "protocol": 120,
    "limits": 121,
    "spawn": 122,
    "exit_watch": 123,
    "pipes": 124,
    "monitor": 125,
    "termination": 126,
    "cleanup": 127,
    "termination_term": 128,
    "termination_grace": 129,
    "termination_kill": 130,
    "termination_wait": 131,
    "termination_term_eperm": 132,
    "termination_term_other_os": 133,
    "termination_term_missing_api": 134,
    "termination_term_non_os": 135,
    "termination_kill_eperm": 136,
    "termination_kill_other_os": 137,
    "termination_kill_missing_api": 138,
    "termination_kill_non_os": 139,
}


class _ExitObserver:
    """Observe exit without reaping: Darwin kqueue or POSIX waitid where exposed."""

    def __init__(self, child):
        self.child = child
        self.queue = None
        self.done = False
        if sys.platform == "darwin":
            self.queue = select.kqueue()
            event = select.kevent(
                child.pid, filter=select.KQ_FILTER_PROC,
                flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT, fflags=select.KQ_NOTE_EXIT,
            )
            try:
                self._events(self.queue.control([event], 1, 0))
            except BaseException:
                self.queue.close()
                raise
        elif not hasattr(os, "waitid"):
            raise ValueError("native exit observation unavailable")

    def _events(self, events):
        for event in events:
            if event.flags & select.KQ_EV_ERROR:
                raise ValueError("native process exit watch rejected")
            if event.ident != self.child.pid or not event.fflags & select.KQ_NOTE_EXIT:
                raise ValueError("native process exit observation differs")
            self.done = True

    def exited(self):
        if self.queue is not None:
            if not self.done:
                self._events(self.queue.control(None, 1, 0))
            return self.done
        return os.waitid(os.P_PID, self.child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None

    def close(self):
        if self.queue is not None:
            self.queue.close()


class _DarwinGroupQuery:
    """Fixed-size libproc reads for the one still-owned process group."""

    def __init__(self):
        import ctypes

        self.ctypes = ctypes
        system = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        self.sysctl = system.sysctlbyname
        self.sysctl.argtypes = (ctypes.c_char_p, ctypes.c_void_p,
                                ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p,
                                ctypes.c_size_t)
        self.sysctl.restype = ctypes.c_int
        library = ctypes.CDLL("/usr/lib/libproc.dylib")
        self.listpids = library.proc_listpids
        self.listpids.argtypes = (ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int)
        self.listpids.restype = ctypes.c_int
        self.pidinfo = library.proc_pidinfo
        self.pidinfo.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                                 ctypes.c_void_p, ctypes.c_int)
        self.pidinfo.restype = ctypes.c_int

    def version(self):
        c = self.ctypes
        buffer = c.create_string_buffer(32)
        size = c.c_size_t(len(buffer))
        if self.sysctl(b"kern.osproductversion", buffer, c.byref(size), None, 0) != 0:
            return None
        raw = buffer.raw[:size.value]
        return raw[:-1] if 1 < size.value <= 32 and raw.endswith(b"\0") else None

    def members(self, pgid):
        c = self.ctypes
        # XNU allocates min(nprocs + 20, supplied capacity) slots even for a
        # group filter. Its NULL-buffer call returns that global allocation
        # estimate without enumerating processes. Require it to cover our
        # entire fixed buffer on both sides of the selection; otherwise a
        # short result might conceal internal truncation. A full buffer may
        # also be silently truncated. Zero can mean a libproc error.
        buffer = (c.c_int32 * 65)()
        capacity = c.sizeof(buffer)
        if self.listpids(2, pgid, None, 0) < capacity:
            return None
        size = self.listpids(2, pgid, buffer, c.sizeof(buffer))
        if (size <= 0 or size >= capacity or size % c.sizeof(c.c_int32)
                or self.listpids(2, pgid, None, 0) < capacity):
            return None
        return tuple(buffer[:size // c.sizeof(c.c_int32)])

    def status(self, pid):
        c = self.ctypes

        class ShortBsdInfo(c.Structure):
            _fields_ = [("pid", c.c_uint32), ("ppid", c.c_uint32),
                        ("pgid", c.c_uint32), ("status", c.c_uint32),
                        ("comm", c.c_char * 16), ("flags", c.c_uint32),
                        ("uid", c.c_uint32), ("gid", c.c_uint32),
                        ("ruid", c.c_uint32), ("rgid", c.c_uint32),
                        ("svuid", c.c_uint32), ("svgid", c.c_uint32),
                        ("reserved", c.c_uint32)]

        if c.sizeof(ShortBsdInfo) != 64:
            return None
        info = ShortBsdInfo()
        # arg=1 permits XNU to find an unreaped zombie. Zero/short return
        # includes denied reads and must never become empty-group evidence.
        if self.pidinfo(pid, 13, 1, c.byref(info), c.sizeof(info)) != 64:
            return None
        return info.pid, info.ppid, info.pgid, info.status


def _darwin_empty_owned_group(child, observer, query=None):
    """Conservative EPERM recovery while the leader is still our zombie."""
    if (sys.platform != "darwin" or os.uname().machine != "arm64"
            or child is None or observer is None or child.returncode is not None
            or not 1 < child.pid < 2**31 or getattr(observer, "child", None) is not child):
        return False
    try:
        if not observer.exited():
            return False
        deadline = time.monotonic() + 1
        query = query if query is not None else _DarwinGroupQuery()
        # Only the reviewed macOS build uses this ABI/behavior. An upgrade
        # needs renewed primary-source and runtime-capture review.
        if query.version() != b"27.0.1" or time.monotonic() > deadline:
            return False
        previous = None
        for _ in range(2):
            members = query.members(child.pid)
            if (time.monotonic() > deadline or type(members) is not tuple
                    or not 1 <= len(members) <= 64 or child.pid not in members
                    or len(set(members)) != len(members)
                    or any(type(pid) is not int or not 1 < pid < 2**31 for pid in members)
                    or previous is not None and set(members) != previous):
                return False
            for pid in members:
                record = query.status(pid)
                if (time.monotonic() > deadline or type(record) is not tuple
                        or len(record) != 4 or any(type(value) is not int for value in record)
                        or record[0] != pid or record[2] != child.pid or record[3] != 5
                        or pid == child.pid and record[1] != os.getpid()):
                    return False
            previous = set(members)
        return observer.exited() and child.returncode is None and time.monotonic() <= deadline
    except Exception:
        return False


def _signal_owned_group(pgid, sig, phase, child=None, observer=None):
    # Keep the first failed operation and a closed error class. An EPERM can
    # also mean an unauthorized live member, so it is never treated as success.
    operation = phase["name"] if phase is not None else None
    signal_group = getattr(os, "killpg", None)
    if not callable(signal_group):
        if phase is not None:
            phase["name"] = operation + "_missing_api"
        raise RuntimeError
    try:
        signal_group(pgid, sig)
    except ProcessLookupError:
        pass
    except PermissionError as failure:
        if phase is not None:
            phase["name"] = operation + ("_eperm" if failure.errno == errno.EPERM else "_other_os")
        if failure.errno == errno.EPERM and _darwin_empty_owned_group(child, observer):
            if phase is not None:
                phase["name"] = operation
            return
        raise
    except OSError:
        if phase is not None:
            phase["name"] = operation + "_other_os"
        raise
    except BaseException:
        if phase is not None:
            phase["name"] = operation + "_non_os"
        raise


def _terminate(child, grace, observer=None, phase=None):
    # Do not poll/reap before signalling the still-owned group: its leader PID
    # cannot be recycled while retained as our unreaped child.
    if phase is not None:
        phase["name"] = "termination_term"
    _signal_owned_group(child.pid, signal.SIGTERM, phase, child, observer)
    if phase is not None:
        phase["name"] = "termination_grace"
    deadline = time.monotonic() + grace
    # Both observation implementations leave the child unreaped.
    while time.monotonic() < deadline:
        if observer is not None:
            try:
                if observer.exited():
                    break
            except Exception:
                observer = None
        time.sleep(min(.02, max(0, deadline - time.monotonic())))
    if phase is not None:
        phase["name"] = "termination_kill"
    _signal_owned_group(child.pid, signal.SIGKILL, phase, child, observer)
    if phase is not None:
        phase["name"] = "termination_wait"
    try:
        return child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        return None


def _lifecycle_canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _lifecycle_source(path, digest, maximum):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or before.st_mode & 0o022 or not 0 < before.st_size <= maximum):
            raise ValueError("native lifecycle source custody differs")
        raw = bytearray()
        while len(raw) <= maximum:
            chunk = os.read(fd, min(8192, maximum + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if (identity(before) != identity(os.fstat(fd))
                or identity(before) != identity(os.stat(path, follow_symlinks=False))
                or len(raw) != before.st_size or hashlib.sha256(raw).hexdigest() != digest):
            raise ValueError("native lifecycle source bytes differ")
        return bytes(raw)
    finally:
        os.close(fd)


class _LifecycleChannel:
    """Trusted-only opt-in. ML children inherit just the progress writer."""

    def __init__(self, spec, journal_fd, key_fd):
        try:
            key = os.read(key_fd, 33)
        finally:
            os.close(key_fd)
        unsigned = dict(spec)
        tag = unsigned.pop("lifecycle_tag", None)
        if (len(key) != 32 or type(tag) is not str or len(tag) != 64
                or any(char not in "0123456789abcdef" for char in tag)
                or not hmac.compare_digest(tag, hmac.new(
                    key, b"synaptic-native-lifecycle-launch/v1\0" + _lifecycle_canonical(unsigned),
                    hashlib.sha256).hexdigest())):
            raise ValueError("native lifecycle launch authentication differs")
        invocation, selection = spec["invocation"], spec["lifecycle"]
        expected = {
            "argv": [invocation["executable"], "-I", "-S", "-B", invocation["worker"],
                     "--request", invocation["request"]],
            "cwd": invocation["private_root"],
            "environment": {"PATH": os.path.dirname(invocation["executable"]) + ":/usr/bin:/bin",
                            "HOME": invocation["private_root"], "TMPDIR": invocation["private_root"],
                            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"},
            **invocation["limits"], "invocation": invocation, "lifecycle": selection,
            "invocation_digest": hashlib.sha256(_lifecycle_canonical(invocation)).hexdigest(),
        }
        if (_lifecycle_canonical(unsigned) != _lifecycle_canonical(expected)
                or invocation["guardian"] != os.path.abspath(__file__)):
            raise ValueError("native lifecycle invocation binding differs")
        request_bytes = _lifecycle_source(invocation["request"], invocation["request_digest"], 2 << 20)
        request = json.loads(request_bytes)
        if (_lifecycle_canonical(request) != request_bytes
                or _lifecycle_canonical(request["lifecycle"]) != _lifecycle_canonical(selection)
                or type(selection) is not dict or set(selection) != {"schema", "policy", "protocol_sha256"}
                or selection["schema"] != "synaptic-native-lifecycle-launch/v1"):
            raise ValueError("native lifecycle request binding differs")
        # -I -S has no project import roots. Verify the bound sibling codec
        # bytes before execution, using only already trusted guardian stdlib.
        path = os.path.join(os.path.dirname(__file__), "native_lifecycle_protocol.py")
        raw = _lifecycle_source(path, selection["protocol_sha256"], 65536)
        codec = types.ModuleType("_synaptic_native_lifecycle_protocol")
        sys.modules[codec.__name__] = codec  # dataclasses require the defining module.
        exec(compile(bytes(raw), path, "exec"), codec.__dict__)
        policy = selection["policy"]
        if (type(policy) is not dict or set(policy) != {"schema", "maximum_progress_records", "maximum_journal_bytes"}
                or policy["schema"] != "synaptic-native-lifecycle-policy/v1"):
            raise ValueError("native lifecycle policy differs")
        policy = codec.NativeLifecyclePolicyV1(policy["maximum_progress_records"], policy["maximum_journal_bytes"])
        if policy.maximum_journal_bytes > spec["maximum_file_bytes"]:
            raise ValueError("native lifecycle file budget differs")
        self.codec = codec
        self.writer = codec.NativeLifecycleWriter(journal_fd, invocation_digest=spec["invocation_digest"],
                                                   policy=policy, key=key)
        self.journal_fd = journal_fd
        self.read_fd, self.write_fd = os.pipe()
        self.pending = bytearray()

    def spawned(self):
        os.close(self.write_fd)
        self.write_fd = None

    def consume(self, chunk):
        self.pending.extend(chunk)
        while b"\n" in self.pending:
            end = self.pending.index(b"\n")
            if end >= self.codec.MAXIMUM_RECORD_BYTES:
                raise ValueError("native progress record limit")
            event = self.codec.parse_worker_progress(bytes(self.pending[:end]))
            self.writer.append(event)
            del self.pending[:end + 1]
        if len(self.pending) >= self.codec.MAXIMUM_RECORD_BYTES:
            raise ValueError("native progress partial record limit")

    def finish(self, outcome):
        self.writer.append({"kind": "terminal", "outcome": outcome})

    def close(self):
        for name in ("read_fd", "write_fd", "journal_fd"):
            fd = getattr(self, name, None)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    # The process is exiting and the worker is already cleaned
                    # up; descriptor cleanup must not emit an exception trace.
                    pass
                setattr(self, name, None)


def supervise(spec, control_fd, phase=None, *, lifecycle=None):
    if phase is None:
        phase = {"name": "limits"}
    phase["name"] = "limits"
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # This limits each file; the worker/artifact inventory enforces aggregate size.
    maximum_file = spec["maximum_file_bytes"]
    resource.setrlimit(resource.RLIMIT_FSIZE, (maximum_file, maximum_file))
    child = None
    observer = None
    selector = selectors.DefaultSelector()
    started = time.monotonic()
    reason = "worker_exited"
    counts = {"stdout": 0, "stderr": 0}
    try:
        phase["name"] = "spawn"
        argv = spec["argv"]
        extra = {}
        if lifecycle is not None:
            argv = [*argv, "--progress-fd", str(lifecycle.write_fd)]
            extra["pass_fds"] = (lifecycle.write_fd,)
        child = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, cwd=spec["cwd"], env=spec["environment"],
            start_new_session=True, close_fds=True, **extra,
        )
        if lifecycle is not None:
            lifecycle.spawned()
        phase["name"] = "exit_watch"
        observer = _ExitObserver(child)
        # Only the guardian owns this live handle. Parent records refer to the
        # guardian; they cannot reacquire this child's cancellation authority.
        phase["name"] = "pipes"
        for pipe, name in ((child.stdout, "stdout"), (child.stderr, "stderr")):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, name)
        os.set_blocking(control_fd, False)
        selector.register(control_fd, selectors.EVENT_READ, "control")
        if lifecycle is not None:
            os.set_blocking(lifecycle.read_fd, False)
            selector.register(lifecycle.read_fd, selectors.EVENT_READ, "progress")
        phase["name"] = "monitor"
        while True:
            if time.monotonic() - started >= spec["timeout_seconds"]:
                reason = "deadline"
                break
            # Observe without reaping, retaining PID ownership until group cleanup.
            exited = observer.exited()
            events = selector.select(.02)
            for key, _ in events:
                chunk = os.read(key.fd, 4096 if key.data == "progress" else 65536)
                if key.data == "control":
                    # Any parent command requests cancellation; EOF means owner loss.
                    reason = "cancel_requested" if chunk else "owner_lost"
                    break
                if key.data == "progress":
                    if not chunk:
                        selector.unregister(key.fileobj)
                        if lifecycle.pending:
                            raise ValueError("native progress incomplete record")
                    else:
                        lifecycle.consume(chunk)
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                else:
                    counts[key.data] += len(chunk)
                    if counts[key.data] > spec["maximum_output_bytes"]:
                        reason = "output_limit"
                        break
            if reason != "worker_exited":
                break
            if exited:
                # Drain to EOF with the same limits. A descendant retaining a
                # pipe cannot keep this loop alive past the overall deadline.
                if all(key.data == "control" for key in selector.get_map().values()):
                    break
        phase["name"] = "termination"
        code = _terminate(child, spec["termination_grace_seconds"], observer, phase)
        return {
            "schema": "synaptic-native-process-outcome/v1",
            "reason": reason, "exit_code": code, "termination_confirmed": code is not None,
            "stdout_bytes": counts["stdout"], "stderr_bytes": counts["stderr"],
        }
    finally:
        # Preserve the failed phase when cleanup follows an earlier exception.
        if sys.exc_info()[0] is None:
            phase["name"] = "cleanup"
        selector.close()
        if child is not None:
            # _terminate already reaps on normal return. Do not signal a reused
            # process ID after reaping; this finally only owns an unreaped child.
            if child.returncode is None:
                _terminate(child, spec["termination_grace_seconds"], observer)
            child.stdout.close()
            child.stderr.close()
        if observer is not None:
            observer.close()


def main():
    # Descriptor and JSON come only from the admitted launcher, not CLI users.
    phase = {"name": "protocol"}
    lifecycle = None
    try:
        if len(sys.argv) not in (2, 4):
            return _FAILURE_CODES["protocol"]
        control_fd = int(sys.argv[1])
        raw = sys.stdin.buffer.readline(65537)
        if len(raw) > 65536 or not raw.endswith(b"\n"):
            return _FAILURE_CODES["protocol"]
        spec = json.loads(raw)
        if len(sys.argv) == 4:
            if _lifecycle_canonical(spec) + b"\n" != raw:
                return _FAILURE_CODES["protocol"]
            journal_fd, key_fd = int(sys.argv[2]), int(sys.argv[3])
            if min(control_fd, journal_fd, key_fd) < 3 or len({control_fd, journal_fd, key_fd}) != 3:
                return _FAILURE_CODES["protocol"]
            lifecycle = _LifecycleChannel(spec, journal_fd, key_fd)
        elif "lifecycle" in spec:
            return _FAILURE_CODES["protocol"]
        outcome = (supervise(spec, control_fd, phase) if lifecycle is None
                   else supervise(spec, control_fd, phase, lifecycle=lifecycle))
        if lifecycle is not None:
            lifecycle.finish(outcome)
        phase["name"] = "protocol"
        sys.stdout.write(json.dumps(outcome, sort_keys=True, separators=(",", ":")))
        return 0
    except BaseException:
        # No exception values, library stderr or credential-bearing diagnostics.
        return _FAILURE_CODES[phase["name"]]
    finally:
        if lifecycle is not None:
            lifecycle.close()


if __name__ == "__main__":
    raise SystemExit(main())
