"""Bounded POSIX file reads for native macOS candidates.

Uses openat/dir_fd and O_NOFOLLOW, available on supported Darwin and Linux.
There is no /proc, openat2, pidfd, or claim of protection from a malicious
process running as the same uid. The host retains exclusive mutation ownership.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat


def file_identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def relative_member(value: str) -> tuple[str, ...]:
    if type(value) is not str or not value or len(value.encode()) > 4096:
        raise ValueError("native member path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(p in (".", "..") for p in path.parts):
        raise ValueError("native member path is noncanonical")
    if any(ord(c) < 32 for c in value) or "\\" in value:
        raise ValueError("native member path is invalid")
    return path.parts


@dataclass(frozen=True)
class NativeFile:
    path: str
    size: int
    sha256: str

    def __post_init__(self):
        relative_member(self.path)
        if type(self.size) is not int or not 0 <= self.size <= 512 * 1024 * 1024:
            raise ValueError("native file size is invalid")
        if (type(self.sha256) is not str or len(self.sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.sha256)):
            raise ValueError("native file digest is invalid")


class NativeTree:
    """Retain all ancestors; reject redirected, non-owned or writable roots.

    Ordinary runtime trees may have mode 0755. Output/input private trees must
    have no group/other permission. An OS-owned ancestor is allowed, but the
    selected tree itself must belong to this uid.
    """

    def __init__(self, path: Path, *, private: bool = False):
        if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
            raise ValueError("native descriptor traversal is unavailable")
        if not path.is_absolute() or str(path) != os.path.normpath(str(path)):
            raise ValueError("native root must be canonical absolute")
        self.path = path
        self._private = private
        self._stack = ExitStack()
        self._chain = []
        try:
            parent = None
            for part in ("/", *path.parts[1:]):
                fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                self._stack.callback(os.close, fd)
                info = os.fstat(fd)
                if info.st_uid not in (0, os.getuid()) or info.st_mode & 0o022:
                    # /tmp is intentionally not accepted as a native runtime root.
                    raise ValueError("native root ancestor is writable or unowned")
                self._chain.append((parent, part, fd, (info.st_dev, info.st_ino)))
                parent = fd
            self.fd = parent
            info = os.fstat(self.fd)
            if info.st_uid != os.getuid() or (private and info.st_mode & 0o077):
                raise ValueError("native root ownership or private mode differs")
        except BaseException:
            self.close()
            raise

    def check(self):
        for parent, name, fd, expected in self._chain:
            current = os.stat(name, dir_fd=parent, follow_symlinks=False)
            held = os.fstat(fd)
            if (not stat.S_ISDIR(current.st_mode)
                    or (current.st_dev, current.st_ino) != expected
                    or (held.st_dev, held.st_ino) != expected
                    or current.st_uid not in (0, os.getuid())
                    or current.st_mode & 0o022):
                raise ValueError("native directory changed")
        leaf = os.fstat(self.fd)
        if leaf.st_uid != os.getuid() or (self._private and leaf.st_mode & 0o077):
            raise ValueError("native private root permissions changed")

    def close(self):
        self._stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def read(self, member: str, *, maximum: int = 512 * 1024 * 1024) -> bytes:
        parts = relative_member(member)
        self.check()
        with ExitStack() as stack:
            parent = self.fd
            chain = []
            for part in parts[:-1]:
                fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                stack.callback(os.close, fd)
                info = os.fstat(fd)
                if info.st_uid != os.getuid() or info.st_mode & 0o022:
                    raise ValueError("native member ancestor is writable or unowned")
                chain.append((parent, part, fd, (info.st_dev, info.st_ino)))
                parent = fd
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            stack.callback(os.close, fd)
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                    or before.st_mode & 0o022 or before.st_size > maximum):
                raise ValueError("native member is not an owned bounded regular file")
            chunks = []
            size = 0
            while size <= maximum:
                chunk = os.read(fd, min(1024 * 1024, maximum + 1 - size))
                if not chunk:
                    break
                size += len(chunk)
                chunks.append(chunk)
            current = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
            if (file_identity(before) != file_identity(os.fstat(fd))
                    or file_identity(before) != file_identity(current)
                    or size != before.st_size or size > maximum):
                raise ValueError("native member changed during read")
            for ancestor, name, held, identity in chain:
                actual = os.stat(name, dir_fd=ancestor, follow_symlinks=False)
                if (not stat.S_ISDIR(actual.st_mode)
                        or (actual.st_dev, actual.st_ino) != identity
                        or (os.fstat(held).st_dev, os.fstat(held).st_ino) != identity):
                    raise ValueError("native member directory changed")
            self.check()
            return b"".join(chunks)

    def write_exclusive(self, member: str, raw: bytes, *, maximum: int = 1024 * 1024):
        """Publish bounded direct-child bytes; retain the 1 MiB default.

        Larger callers must supply their admitted limit, within the existing
        native per-file ceiling. Reject before creating any file, and use the
        same limit for the authenticated same-tree readback.
        """
        if (type(maximum) is not int or not 1 <= maximum <= 512 * 1024 * 1024
                or len(relative_member(member)) != 1 or type(raw) is not bytes or len(raw) > maximum):
            raise ValueError("native publication must be bounded direct-child bytes")
        self.check()
        fd = os.open(member, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=self.fd)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        self.check()
        if self.read(member, maximum=maximum) != raw:
            raise ValueError("native publication readback differs")

    def members(self, *, aliases: dict[str, str] | None = None) -> tuple[str, ...]:
        """List every regular file, including importable bytecode and startup files."""
        self.check()
        result = []
        aliases = aliases or {}
        seen_aliases = set()

        def walk(fd, prefix, depth):
            if depth > 32:
                raise ValueError("native inventory depth exceeded")
            for name in sorted(os.listdir(fd)):
                member = f"{prefix}/{name}" if prefix else name
                relative_member(member)
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if member in aliases:
                    if (not stat.S_ISLNK(info.st_mode) or info.st_uid != os.getuid()
                            or os.readlink(name, dir_fd=fd) != aliases[member]):
                        raise ValueError("native declared alias differs")
                    seen_aliases.add(member)
                elif stat.S_ISDIR(info.st_mode):
                    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    try:
                        if (info.st_dev, info.st_ino) != (os.fstat(child).st_dev, os.fstat(child).st_ino):
                            raise ValueError("native inventory directory changed")
                        if info.st_uid != os.getuid() or info.st_mode & 0o022:
                            raise ValueError("native inventory directory is writable or unowned")
                        walk(child, member, depth + 1)
                    finally:
                        os.close(child)
                elif stat.S_ISREG(info.st_mode):
                    result.append(member)
                    if len(result) > 100000:
                        raise ValueError("native inventory member count exceeded")
                else:
                    raise ValueError("native inventory contains a nonregular member")
        walk(self.fd, "", 0)
        if seen_aliases != set(aliases):
            raise ValueError("native declared alias is absent")
        self.check()
        return tuple(sorted(result))

    def inventory(self, *, maximum_total: int = 4 * 1024**3) -> tuple[NativeFile, ...]:
        names = self.members()
        result = []
        total = 0
        for name in names:
            raw = self.read(name)
            total += len(raw)
            if total > maximum_total:
                raise ValueError("native inventory byte budget exceeded")
            result.append(NativeFile(name, len(raw), hashlib.sha256(raw).hexdigest()))
        if self.members() != names:
            raise ValueError("native inventory membership changed")
        return tuple(result)
