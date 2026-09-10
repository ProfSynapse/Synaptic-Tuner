"""Linux process-group ownership for locally spawned runtimes."""

from __future__ import annotations

import math
import os
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any

__all__: list[str] = []

_TOKEN = object()
_UNKNOWN = object()
_MAX_TIMEOUT = 300.0
_MAX_PROC_ENTRIES = 131_072
_MAX_STAT_BYTES = 4096


class OwnedProcessError(RuntimeError):
    """A process family could not be safely owned or stopped."""


class OwnedProcessLease:
    """Ownership of one newly spawned Linux session and process group."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        pgid: int,
        start_time: int | None,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _TOKEN:
            raise TypeError("OwnedProcessLease instances must be created by spawn")
        self._process = process
        self._pgid = pgid
        self._start_time = start_time
        self._pending = True
        self._uncertain = False

    @classmethod
    def spawn(
        cls,
        argv: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        stdin: int | IO[Any] | None = subprocess.DEVNULL,
        stdout: int | IO[Any] | None = subprocess.DEVNULL,
        stderr: int | IO[Any] | None = subprocess.DEVNULL,
    ) -> "OwnedProcessLease":
        _require_host()
        process = subprocess.Popen(
            _argv(argv),
            cwd=_cwd(cwd),
            env=_environment(environment),
            shell=False,
            start_new_session=True,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
        )
        try:
            identity = _identity(process.pid)
            if identity is None or identity is _UNKNOWN or identity[1] != process.pid:
                raise OwnedProcessError("spawned process group identity was not established")
            return cls(process, process.pid, identity[2], _token=_TOKEN)
        except BaseException as error:
            # start_new_session establishes PGID == PID before exec. Retain the
            # unreaped leader as an anchor even when procfs parsing failed.
            error.cleanup_lease = cls(  # type: ignore[attr-defined]
                process, process.pid, None, _token=_TOKEN
            )
            raise

    @property
    def pid(self) -> int:
        return self._process.pid

    @property
    def cleanup_pending(self) -> bool:
        return self._pending

    def close(
        self, *, term_timeout: float = 5.0, kill_timeout: float = 5.0
    ) -> bool:
        term = _timeout(term_timeout)
        kill = _timeout(kill_timeout)
        if not self._pending:
            return True
        if self._uncertain or not self._anchored():
            self._uncertain = True
            return False
        if not self._signal(signal.SIGTERM):
            return not self._pending
        if self._wait(term):
            return True
        if not self._anchored():
            self._uncertain = True
            return False
        if not self._signal(signal.SIGKILL):
            return not self._pending
        return self._wait(kill)

    def _anchored(self) -> bool:
        identity = _identity(self.pid)
        if identity is _UNKNOWN:
            return False
        if self._start_time is None:
            return identity is not None and identity[1] == self._pgid
        return (
            identity is not None
            and identity[1] == self._pgid
            and identity[2] == self._start_time
        )

    def _signal(self, sig: int) -> bool:
        try:
            os.killpg(self._pgid, sig)
            return True
        except ProcessLookupError:
            members = self._members(time.monotonic() + 0.1)
            if members is None or members:
                self._uncertain = True
            else:
                self._finish()
            return False
        except (PermissionError, OSError):
            self._uncertain = True
            return False

    def _members(self, deadline: float) -> tuple[int, ...] | None:
        result: list[int] = []
        count = 0
        try:
            with os.scandir("/proc") as entries:
                while True:
                    if time.monotonic() >= deadline or count >= _MAX_PROC_ENTRIES:
                        return None
                    try:
                        entry = next(entries)
                    except StopIteration:
                        return tuple(result)
                    count += 1
                    if not entry.name.isdecimal():
                        continue
                    identity = _identity(int(entry.name))
                    if identity is _UNKNOWN:
                        return None
                    if (
                        identity is not None
                        and identity[1] == self._pgid
                        and identity[0] != "Z"
                    ):
                        result.append(int(entry.name))
        except OSError:
            return None

    def _wait(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            members = self._members(deadline)
            if members == ():
                self._finish()
                return not self._pending
            if members is None or self._uncertain or time.monotonic() >= deadline:
                return False
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))

    def _finish(self) -> None:
        try:
            self._process.wait(timeout=0)
        except (subprocess.TimeoutExpired, ChildProcessError):
            self._uncertain = True
            return
        self._pending = False

    def __enter__(self) -> "OwnedProcessLease":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        try:
            stopped = self.close()
        except BaseException:
            if exc_type is not None:
                return False
            raise
        if not stopped and exc_type is None:
            raise OwnedProcessError("owned process family cleanup remains unresolved")
        return False


def _require_host() -> None:
    if (
        os.name != "posix"
        or not Path("/proc/self/stat").is_file()
        or not hasattr(os, "killpg")
    ):
        raise OwnedProcessError(
            "owned process families require Linux procfs and POSIX process groups"
        )


def _identity(pid: int) -> tuple[str, int, int] | None | object:
    try:
        with open(f"/proc/{pid}/stat", "rb", buffering=0) as stream:
            raw = stream.read(_MAX_STAT_BYTES + 1)
        if len(raw) > _MAX_STAT_BYTES:
            return _UNKNOWN
        fields = raw.decode("ascii").rsplit(") ", 1)[1].split()
        return fields[0], int(fields[2]), int(fields[19])
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, ValueError, IndexError):
        return _UNKNOWN


def _argv(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not value:
        raise TypeError("argv must be a non-empty sequence")
    result = tuple(value)
    if any(type(item) is not str or not item or "\0" in item for item in result):
        raise TypeError("invalid argv")
    return result


def _cwd(value: Path) -> str:
    if not isinstance(value, Path) or not value.is_absolute() or not value.is_dir():
        raise TypeError("cwd must be an existing absolute Path")
    return str(value)


def _environment(value: Mapping[str, str]) -> dict[str, str]:
    if type(value) is not dict:
        raise TypeError("environment must be an explicit dict")
    result = dict(value)
    if any(
        type(key) is not str
        or type(item) is not str
        or not key
        or "=" in key
        or "\0" in key + item
        for key, item in result.items()
    ):
        raise TypeError("invalid environment")
    return result


def _timeout(value: float) -> float:
    if (
        type(value) not in (int, float)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0 <= value <= _MAX_TIMEOUT
    ):
        raise TypeError("timeouts must be finite numbers from zero through 300 seconds")
    return float(value)
