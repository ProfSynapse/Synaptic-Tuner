"""Standalone stdlib guardian for one owned native worker, started with -I -S.

The parent holds a pipe open. EOF (including parent death) terminates the worker
process group. A persisted PID is never reused as restart-time signal authority.
This is POSIX/Darwin process supervision, not a sandbox or Linux pidfd guarantee.
"""

from __future__ import annotations

import json
import os
import resource
import select
import selectors
import signal
import subprocess
import sys
import time


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


def _terminate(child, grace, observer=None):
    # Do not poll/reap before signalling the still-owned group: its leader PID
    # cannot be recycled while retained as our unreaped child.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
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
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        return child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        return None


def supervise(spec, control_fd, phase=None):
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
        child = subprocess.Popen(
            spec["argv"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, cwd=spec["cwd"], env=spec["environment"],
            start_new_session=True, close_fds=True,
        )
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
        phase["name"] = "monitor"
        while True:
            if time.monotonic() - started >= spec["timeout_seconds"]:
                reason = "deadline"
                break
            # Observe without reaping, retaining PID ownership until group cleanup.
            exited = observer.exited()
            events = selector.select(.02)
            for key, _ in events:
                chunk = os.read(key.fd, 65536)
                if key.data == "control":
                    # Any parent command requests cancellation; EOF means owner loss.
                    reason = "cancel_requested" if chunk else "owner_lost"
                    break
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
        code = _terminate(child, spec["termination_grace_seconds"], observer)
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
    try:
        if len(sys.argv) != 2:
            return _FAILURE_CODES["protocol"]
        control_fd = int(sys.argv[1])
        raw = sys.stdin.buffer.readline(65537)
        if len(raw) > 65536 or not raw.endswith(b"\n"):
            return _FAILURE_CODES["protocol"]
        spec = json.loads(raw)
        outcome = supervise(spec, control_fd, phase)
        phase["name"] = "protocol"
        sys.stdout.write(json.dumps(outcome, sort_keys=True, separators=(",", ":")))
        return 0
    except BaseException:
        # No exception values, library stderr or credential-bearing diagnostics.
        return _FAILURE_CODES[phase["name"]]


if __name__ == "__main__":
    raise SystemExit(main())
