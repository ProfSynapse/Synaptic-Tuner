"""Caller-pumped native guardian sessions; public lifecycle capabilities stay off.

Store operations remain on the constructing thread. Only a live owner's pipe
authorizes stop; cold records never authorize signals or another submission.
This transport primitive does not yet provide authenticated restart recovery.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import os
from pathlib import Path
import select
import subprocess
import threading
import time

from tuner.runtime.native_files import NativeTree
from tuner.runtime.native_mlx_material import canonical, strict_json
from tuner.runtime.native_process import NativeInvocation, NativeOwnedProcess


@dataclass
class _Active:
    invocation: NativeInvocation
    guardian: object
    control: int
    root: NativeTree
    started: float
    output: bytearray = field(default_factory=bytearray)
    eof: bool = False
    stopping: bool = False


class NativeProcessSession:
    """Additive asynchronous transport; no worker thread or private database."""

    def __init__(self, records):
        self._records = NativeOwnedProcess(records)
        self._pid, self._thread = os.getpid(), threading.get_ident()
        self._live = {}
        self._lock = threading.Lock()

    def _owner(self):
        if os.getpid() != self._pid or threading.get_ident() != self._thread:
            raise ValueError("native session requires its owning thread")

    def __enter__(self):
        self._owner()
        return self

    def __exit__(self, *_):
        self.close()

    def _release(self, active):
        with self._lock:
            self._live.pop(active.invocation.attempt, None)
            os.close(active.control)
        guardian = active.guardian
        try:
            if guardian.stdin and not guardian.stdin.closed:
                guardian.stdin.close()
            try:
                guardian.wait(timeout=2 * active.invocation.limits.termination_grace_seconds + 2)
            except subprocess.TimeoutExpired:
                # The guardian still owns cleanup. Never signal its persisted PID.
                pass
        finally:
            if guardian.stdout:
                guardian.stdout.close()
            active.root.close()

    def _fail(self, active, category):
        try:
            self._records._retain(active.invocation, "diagnostic", {"category": category})
        finally:
            self._release(active)
        raise ValueError("native session outcome is uncertain")

    def start_once(self, invocation):
        self._owner()
        if type(invocation) is not NativeInvocation or os.name != "posix":
            raise TypeError("native session requires an exact POSIX invocation")
        root = NativeTree(Path(invocation.private_root), private=True)
        read_fd = write_fd = None
        active = None
        try:
            for name in ("executable", "guardian", "worker", "request"):
                path = Path(getattr(invocation, name))
                with NativeTree(path.parent) as source:
                    raw = source.read(path.name, maximum=64 << 20)
                if hashlib.sha256(raw).hexdigest() != getattr(invocation, name + "_digest"):
                    raise ValueError("native session input differs")
            root.check()
            self._records._retain(invocation, "claim", {"automatic_replay": False})
            read_fd, write_fd = os.pipe()
            os.set_blocking(write_fd, False)
            environment = {
                "PATH": str(Path(invocation.executable).parent) + ":/usr/bin:/bin",
                "HOME": invocation.private_root, "TMPDIR": invocation.private_root,
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
            spec = canonical({
                "argv": [invocation.executable, "-I", "-S", "-B", invocation.worker,
                         "--request", invocation.request],
                "cwd": invocation.private_root, "environment": environment,
                **asdict(invocation.limits),
            }) + b"\n"
            if len(spec) > 65536:
                raise ValueError("native session protocol exceeds bound")
            guardian = subprocess.Popen(
                [invocation.executable, "-I", "-S", "-B", invocation.guardian, str(read_fd)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                env=environment, cwd=invocation.private_root, start_new_session=True,
                close_fds=True, pass_fds=(read_fd,),
            )
            os.close(read_fd)
            read_fd = None
            active = _Active(invocation, guardian, write_fd, root, time.monotonic())
            root = write_fd = None
            with self._lock:
                self._live[invocation.attempt] = active
            self._records._retain(invocation, "spawn", {
                "guardian_pid": guardian.pid, "restart_signal_authority": False,
            })
            os.set_blocking(guardian.stdin.fileno(), False)
            os.set_blocking(guardian.stdout.fileno(), False)
            # Send the bounded protocol without waiting for training. Even a
            # guardian that stops reading cannot block the host indefinitely.
            deadline, offset = time.monotonic() + 5, 0
            while offset < len(spec):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([], [guardian.stdin], [], remaining)[1]:
                    raise ValueError("native session startup deadline")
                try:
                    offset += os.write(guardian.stdin.fileno(), spec[offset:offset + 4096])
                except BlockingIOError:
                    continue
            guardian.stdin.close()
            return {"state": "running", "automatic_replay": False}
        except BaseException:
            if active is not None:
                self._release(active)
            raise
        finally:
            if read_fd is not None:
                os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)
            if root is not None:
                root.close()

    def poll(self, invocation):
        """Nonblocking terminal collection, on the store-owning thread only."""
        self._owner()
        if type(invocation) is not NativeInvocation:
            raise TypeError("native session requires an exact invocation")
        active = self._live.get(invocation.attempt)
        if active is None:
            return self._records.observe(invocation)
        if invocation != active.invocation:
            raise ValueError("native live invocation differs")
        try:
            active.root.check()
            while not active.eof:
                try:
                    chunk = os.read(active.guardian.stdout.fileno(), 4097 - len(active.output))
                except BlockingIOError:
                    break
                if not chunk:
                    active.eof = True
                else:
                    active.output.extend(chunk)
                    if len(active.output) > 4096:
                        return self._fail(active, "guardian_response_limit")
            status = active.guardian.poll()
            if status is None or not active.eof:
                limits = invocation.limits
                if time.monotonic() - active.started > limits.timeout_seconds + 2 * limits.termination_grace_seconds + 10:
                    return self._fail(active, "guardian_response_timeout")
                return {"state": "running", "automatic_replay": False}
            if status != 0:
                return self._fail(active, "guardian_exit")
            outcome = strict_json(bytes(active.output), maximum=4096)
            if (type(outcome) is not dict or set(outcome) != {"schema", "reason", "exit_code",
                    "termination_confirmed", "stdout_bytes", "stderr_bytes"}
                    or outcome["schema"] != "synaptic-native-process-outcome/v1"
                    or outcome["termination_confirmed"] is not True
                    or type(outcome["exit_code"]) is not int
                    or outcome["reason"] not in {"worker_exited", "cancel_requested", "owner_lost", "deadline", "output_limit"}
                    or any(type(outcome[key]) is not int or not 0 <= outcome[key] <= invocation.limits.maximum_output_bytes + 65536
                           for key in ("stdout_bytes", "stderr_bytes"))):
                return self._fail(active, "guardian_response_invalid")
            self._records._retain(invocation, "outcome", outcome)
        except BaseException:
            if self._live.get(invocation.attempt) is active:
                self._release(active)
            raise
        self._release(active)
        return self._records.observe(invocation)

    def observe(self, invocation):
        return self.poll(invocation)

    def cancel(self, invocation):
        """Cross-thread stop request uses only the current live owner's pipe."""
        if type(invocation) is not NativeInvocation or os.getpid() != self._pid:
            return False
        with self._lock:
            active = self._live.get(invocation.attempt)
            if active is None or active.invocation != invocation:
                return False
            if active.stopping:
                return True
            try:
                os.write(active.control, b"stop")
                active.stopping = True
                return True
            except OSError:
                return False

    def close(self):
        """Relinquish live pipes for bounded guardian cleanup; never resubmit."""
        self._owner()
        for active in tuple(self._live.values()):
            self._release(active)
