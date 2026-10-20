"""Bounded, owned POSIX command output for host-side observations.

The caller supplies exact argv and a small environment. Output is drained as
it arrives with one combined stdout/stderr limit; no exception exposes output,
argv, environment, or subprocess exception text.
"""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time


class BoundedCommandError(RuntimeError):
    pass


def run_bounded_command(argv, *, cwd=None, environment=None, input_bytes=b"",
                        maximum=4096, timeout=15):
    if (os.name != "posix" or not hasattr(os, "killpg")
            or not isinstance(argv, (tuple, list)) or not argv
            or any(type(value) is not str or not value for value in argv)
            or type(input_bytes) is not bytes or len(input_bytes) > 1024 * 1024
            or type(maximum) is not int or not 1 <= maximum <= 1024 * 1024
            or type(timeout) not in (int, float) or not 0 < timeout <= 300):
        raise BoundedCommandError("bounded_command_invalid")
    process = None
    selector = selectors.DefaultSelector()
    try:
        process = subprocess.Popen(
            argv, cwd=cwd, env=environment, stdin=subprocess.PIPE if input_bytes else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, close_fds=True,
        )
        deadline = time.monotonic() + timeout
        stdout, total, written = [], 0, 0
        for stream, role in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, role)
        if input_bytes:
            os.set_blocking(process.stdin.fileno(), False)
            selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BoundedCommandError("bounded_command_deadline")
            for key, _ in selector.select(remaining):
                stream, role = key.fileobj, key.data
                if role == "stdin":
                    try:
                        written += os.write(stream.fileno(), input_bytes[written:written + 65536])
                    except BrokenPipeError:
                        written = len(input_bytes)
                    if written == len(input_bytes):
                        selector.unregister(stream)
                        stream.close()
                    continue
                raw = os.read(stream.fileno(), min(65536, maximum + 1 - total))
                if raw:
                    total += len(raw)
                    if total > maximum:
                        raise BoundedCommandError("bounded_command_output_limit")
                    if role == "stdout":
                        stdout.append(raw)
                else:
                    selector.unregister(stream)
                    stream.close()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BoundedCommandError("bounded_command_deadline")
        return process.wait(timeout=remaining), b"".join(stdout)
    except Exception:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        raise BoundedCommandError("bounded_command_failed") from None
    finally:
        selector.close()
        if process is not None:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
