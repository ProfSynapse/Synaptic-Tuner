"""Installed-wheel integrity proof for the packaged-training admission boundary."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import os
import re
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path


PACKAGED_WORKER_CLOSURE_SCHEMA = "synaptic-packaged-training-worker-closure/v1"
PACKAGED_WORKER_CLOSURE_RESOURCE = "manifests/packaged-training-worker-v1.json"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
MAX_RESOURCE_BYTES = 1024 * 1024
PACKAGED_WORKER_MEMBERS = ("packaged_sft_child.py", "packaged_sft_execution.py", "packaged_training_worker.py", "packaged_worker_closure.py", "releases.py")


def stable_read(path: Path, maximum: int = MAX_RESOURCE_BYTES, *, _digest_only=False):
    """Bounded regular-file read with retained ancestors and file identity.

    POSIX opens each component relative to a retained no-follow descriptor.
    Windows retains no-delete-share directory handles, preventing ancestor
    rename while the file descriptor is read and checked.
    """
    path = Path(os.path.abspath(path))
    def identity(info):
        # CPython 3.12 Windows lstat exposes creation time as ctime while
        # fstat exposes change time. Compare each API's change time separately.
        stamp = info.st_birthtime_ns if os.name == "nt" else info.st_ctime_ns
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, stamp)
    def regular(info):
        return (stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                and not getattr(info, "st_file_attributes", 0) & 0x400)
    with ExitStack() as stack:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            create = kernel.CreateFileW
            create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                               wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
            create.restype = wintypes.HANDLE
            close = kernel.CloseHandle
            close.argtypes = (wintypes.HANDLE,)
            for parent in reversed(path.parents):
                handle = create(str(parent), 0x80, 1, None, 3, 0x02200000, None)
                if handle == wintypes.HANDLE(-1).value:
                    raise OSError("directory unavailable")
                stack.callback(close, handle)
                info = parent.lstat()
                if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                    raise OSError("linked directory")
            before = path.lstat()
            import msvcrt
            file_handle = create(str(path), 0x80000000, 1, None, 3, 0x00200000, None)
            if file_handle == wintypes.HANDLE(-1).value:
                raise OSError("file unavailable")
            try:
                descriptor = msvcrt.open_osfhandle(file_handle, os.O_RDONLY | os.O_BINARY)
            except BaseException:
                close(file_handle)
                raise
            current = path.lstat
        else:
            parent_fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            stack.callback(os.close, parent_fd)
            ancestors = []
            for part in path.parts[1:-1]:
                child_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
                ancestors.append((parent_fd, part, child_fd))
                parent_fd = child_fd
                stack.callback(os.close, parent_fd)
            before = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
            current = lambda: os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        stack.callback(os.close, descriptor)
        opened = os.fstat(descriptor)
        if not regular(before) or not regular(opened) or identity(before) != identity(opened) or not 0 <= opened.st_size <= maximum:
            raise OSError("invalid regular file")
        chunks = []
        digest = hashlib.sha256()
        size = 0
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
            if not _digest_only:
                chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        named = current()
        if (size != opened.st_size or size > maximum or not regular(after)
                or not regular(named) or identity(opened) != identity(after)
                or identity(opened) != identity(named)):
            raise OSError("file changed")
        if opened.st_ctime_ns != after.st_ctime_ns or before.st_ctime_ns != named.st_ctime_ns:
            raise OSError("file changed")
        if os.name != "nt":
            for parent_fd, part, child_fd in ancestors:
                held = os.fstat(child_fd)
                named_parent = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
                if not stat.S_ISDIR(named_parent.st_mode) or (held.st_dev, held.st_ino) != (named_parent.st_dev, named_parent.st_ino):
                    raise OSError("ancestor changed")
        return (size, digest.hexdigest()) if _digest_only else raw


def stable_file_digest(path: Path, maximum: int) -> tuple[int, str]:
    """Stream large model/archive members through the same retained read proof."""
    return stable_read(path, maximum, _digest_only=True)


def _resource_bytes(resource, maximum: int) -> bytes:
    if isinstance(resource, Path):
        return stable_read(resource, maximum)
    with resource.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if not raw or len(raw) > maximum:
        raise ValueError("resource exceeds limit")
    return raw


class PackagedWorkerClosureError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PackagedWorkerClosureV1:
    digest: str
    members: tuple[tuple[str, int, str], ...]


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def load_packaged_worker_closure() -> PackagedWorkerClosureV1:
    try:
        raw = _resource_bytes(importlib.resources.files("tuner.runtime").joinpath(
            PACKAGED_WORKER_CLOSURE_RESOURCE
        ), MAX_RESOURCE_BYTES)
        document = json.loads(raw.decode("utf-8"))
        if _canonical(document) + b"\n" != raw or set(document) != {"schema_version", "members", "closure_digest"}:
            raise ValueError
        members = document["members"]
        recorded = document["closure_digest"]
        if document["schema_version"] != PACKAGED_WORKER_CLOSURE_SCHEMA or not isinstance(members, list) or not members or not isinstance(recorded, str) or _DIGEST.fullmatch(recorded) is None:
            raise ValueError
        unsigned = dict(document)
        unsigned.pop("closure_digest")
        if hashlib.sha256(_canonical(unsigned)).hexdigest() != recorded:
            raise ValueError
        parsed: list[tuple[str, int, str]] = []
        for member in members:
            if not isinstance(member, dict) or set(member) != {"path", "size_bytes", "sha256"}:
                raise ValueError
            path, size, digest = member["path"], member["size_bytes"], member["sha256"]
            if not isinstance(path, str) or re.fullmatch(r"[a-z][a-z0-9_]*\.py", path) is None or type(size) is not int or not 1 <= size <= MAX_RESOURCE_BYTES or not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
                raise ValueError
            payload = _resource_bytes(importlib.resources.files("tuner.runtime").joinpath(path), size)
            if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError
            parsed.append((path, size, digest))
        if tuple(path for path, _, _ in parsed) != tuple(sorted(set(path for path, _, _ in parsed))):
            raise ValueError
        if tuple(path for path, _, _ in parsed) != PACKAGED_WORKER_MEMBERS:
            raise ValueError
        return PackagedWorkerClosureV1(recorded, tuple(parsed))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PackagedWorkerClosureError("PACKAGED_WORKER_CLOSURE_REJECTED") from exc


__all__ = ["PACKAGED_WORKER_CLOSURE_SCHEMA", "PackagedWorkerClosureError", "PackagedWorkerClosureV1", "load_packaged_worker_closure"]
