"""Fixed stdlib first-stage loader. Its exact bytes are in the submitted command.

Only after authenticating/copying the complete parent inventory does it use
normal Python package imports. This is separate from the isolated engine tree.
"""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys


def _relative(value):
    p = PurePosixPath(value)
    if (type(value) is not str or p.is_absolute() or p.as_posix() != value
            or not p.parts or any(part in {".", ".."} for part in p.parts)
            or "\\" in value or "\0" in value):
        raise ValueError("HF_BOOTSTRAP_INPUT")
    return p.parts


def _read(root, relative, size):
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    held = []
    try:
        parent = os.open(root, flags)
        held.append(parent)
        chain = []
        parts = _relative(relative)
        for part in parts[:-1]:
            child = os.open(part, flags, dir_fd=parent)
            held.append(child)
            chain.append((parent, part, child))
            parent = child
        before = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_size != size:
            raise ValueError("HF_BOOTSTRAP_MEMBER")
        leaf = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                       dir_fd=parent)
        held.append(leaf)
        opened = os.fstat(leaf)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if identity(opened) != identity(before):
            raise ValueError("HF_BOOTSTRAP_MEMBER")
        raw = bytearray()
        while True:
            chunk = os.read(leaf, min(1024 * 1024, size + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > size:
                raise ValueError("HF_BOOTSTRAP_MEMBER")
        if len(raw) != size or identity(os.fstat(leaf)) != identity(before):
            raise ValueError("HF_BOOTSTRAP_MEMBER")
        if identity(os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)) != identity(before):
            raise ValueError("HF_BOOTSTRAP_MEMBER")
        for owner, name, child in chain:
            actual, expected = os.stat(name, dir_fd=owner, follow_symlinks=False), os.fstat(child)
            if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
                raise ValueError("HF_BOOTSTRAP_MEMBER")
        return bytes(raw)
    finally:
        for fd in reversed(held):
            os.close(fd)


def _write(root, relative, raw, *, private=True):
    parts = _relative(relative)
    held = []
    try:
        parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        held.append(parent)
        for part in parts[:-1]:
            try:
                os.mkdir(part, 0o700, dir_fd=parent)
            except FileExistsError:
                pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=parent)
            held.append(child)
            info = os.fstat(child)
            if private and (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700):
                raise ValueError("HF_BOOTSTRAP_PRIVATE")
            parent = child
        fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                     0o600, dir_fd=parent)
        held.append(fd)
        view = memoryview(raw)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise ValueError("HF_BOOTSTRAP_WRITE")
            view = view[count:]
        if private:
            os.fsync(fd)
            os.fsync(parent)
        # Provider mount writes are verified by readback and later host SDK
        # reads; local fsync is not an HF remote-persistence guarantee.
    finally:
        for fd in reversed(held):
            os.close(fd)


def run(payload):
    """Load only the command-bound parent members from the read-only mount."""
    if len(payload) > 1024 * 1024:
        raise ValueError("HF_BOOTSTRAP_INPUT")
    document = json.loads(payload)
    if set(document) != {"schema_version", "parent_members", "worker"} or document["schema_version"] != "synaptic-hf-source-bootstrap/v1":
        raise ValueError("HF_BOOTSTRAP_INPUT")
    members = document["parent_members"]
    if type(members) is not list or not 1 <= len(members) <= 128:
        raise ValueError("HF_BOOTSTRAP_INPUT")
    parent_root = Path("/workspace/hf-parent")
    parent_root.mkdir(mode=0o700)
    total, seen = 0, set()
    for row in members:
        if type(row) is not dict or set(row) != {"path", "size_bytes", "sha256"}:
            raise ValueError("HF_BOOTSTRAP_INPUT")
        name, size = row["path"], row["size_bytes"]
        _relative(name)
        if name in seen or type(size) is not int or not 0 < size <= 4 * 1024 * 1024:
            raise ValueError("HF_BOOTSTRAP_INPUT")
        seen.add(name)
        total += size
        if total > 16 * 1024 * 1024:
            raise ValueError("HF_BOOTSTRAP_INPUT")
        raw = _read("/source", "parent/" + name, size)
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("HF_BOOTSTRAP_MEMBER")
        _write(parent_root, name, raw)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(parent_root))
    from tuner.execution.providers.hf_jobs_provider_v1.source_worker import execute
    return execute(document["worker"])
