"""Native cross-thread/process serialization for protected image operations."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


LOCK_ACQUIRE_SECONDS = 30
LOCK_CLEANUP_SECONDS = 60
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_REPOSITORY = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+$")
_THREAD_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.Lock] = {}


class ImageOperationLockError(RuntimeError):
    def __init__(self, reason_code: str):
        self.reason_code = reason_code if reason_code in {
            "OPERATION_LOCK_INVALID", "OPERATION_LOCK_TIMEOUT",
            "OPERATION_LOCK_CLEANUP_FAILED",
        } else "OPERATION_LOCK_INVALID"
        super().__init__(self.reason_code)


def operation_lock_key(repository: str, child_digest: str) -> str:
    if not _REPOSITORY.fullmatch(repository) or not _DIGEST.fullmatch(child_digest):
        raise ImageOperationLockError("OPERATION_LOCK_INVALID")
    return hashlib.sha256(
        b"synaptic-hf-training-image-operation/v1\0"
        + repository.encode("ascii") + b"\0" + child_digest.encode("ascii")
    ).hexdigest()


def _lock_root() -> Path:
    return Path(tempfile.gettempdir()) / "synaptic-hf-training-image-operation-locks-v1"


def _thread_lock(key: str) -> threading.Lock:
    with _THREAD_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.Lock())


def _native_try_lock(descriptor: int) -> bool:
    try:
        if os.name == "nt":
            import msvcrt
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (OSError, BlockingIOError):
        return False


def _native_unlock(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(descriptor, fcntl.LOCK_UN)


@contextmanager
def image_operation_lock(
    repository: str, child_digest: str, *,
    acquisition_seconds: int = LOCK_ACQUIRE_SECONDS,
    cleanup_seconds: int = LOCK_CLEANUP_SECONDS,
) -> Iterator[str]:
    if acquisition_seconds != LOCK_ACQUIRE_SECONDS or cleanup_seconds != LOCK_CLEANUP_SECONDS:
        raise ImageOperationLockError("OPERATION_LOCK_INVALID")
    key = operation_lock_key(repository, child_digest)
    local = _thread_lock(key)
    acquisition_deadline = time.monotonic() + acquisition_seconds
    acquired_local = False
    descriptor: int | None = None
    native_locked = False
    try:
        remaining = max(0.0, acquisition_deadline - time.monotonic())
        acquired_local = local.acquire(timeout=remaining)
        if not acquired_local:
            raise ImageOperationLockError("OPERATION_LOCK_TIMEOUT")
        root = _lock_root()
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root_info = root.lstat()
        if root.is_symlink() or not stat.S_ISDIR(root_info.st_mode) or root.resolve(strict=True) != root.absolute():
            raise ImageOperationLockError("OPERATION_LOCK_INVALID")
        path = root / f"{key}.lock"
        if path.is_symlink():
            raise ImageOperationLockError("OPERATION_LOCK_INVALID")
        descriptor = os.open(
            path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"0")
            os.fsync(descriptor)
        lock_info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(lock_info.st_mode) or lock_info.st_size != 1
            or getattr(lock_info, "st_nlink", 1) != 1
        ):
            raise ImageOperationLockError("OPERATION_LOCK_INVALID")
        while time.monotonic() < acquisition_deadline:
            if _native_try_lock(descriptor):
                native_locked = True
                break
            time.sleep(min(0.05, max(0.0, acquisition_deadline - time.monotonic())))
        if not native_locked:
            raise ImageOperationLockError("OPERATION_LOCK_TIMEOUT")
        yield key
    except ImageOperationLockError:
        raise
    except (OSError, ValueError) as exc:
        raise ImageOperationLockError("OPERATION_LOCK_INVALID") from exc
    finally:
        cleanup_deadline = time.monotonic() + cleanup_seconds
        cleanup_failed = False
        if native_locked and descriptor is not None:
            try:
                _native_unlock(descriptor)
            except OSError:
                cleanup_failed = True
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                cleanup_failed = True
        if acquired_local:
            try:
                local.release()
            except RuntimeError:
                cleanup_failed = True
        if time.monotonic() > cleanup_deadline:
            cleanup_failed = True
        if cleanup_failed and not any(item is not None for item in sys.exc_info()):
            raise ImageOperationLockError("OPERATION_LOCK_CLEANUP_FAILED")


__all__ = [
    "ImageOperationLockError", "LOCK_ACQUIRE_SECONDS", "LOCK_CLEANUP_SECONDS",
    "image_operation_lock", "operation_lock_key",
]
