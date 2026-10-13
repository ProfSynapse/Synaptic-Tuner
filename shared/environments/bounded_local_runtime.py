"""Opt-in bounded fixture filesystem; containment, not a process security sandbox.

Only Linux descriptor-relative I/O is supported. Deadlines are cooperative checks,
not interruption of filesystem calls or cancellation of an executing process.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields
import json
import math
import os
from pathlib import Path
import secrets
import stat
import sys
import time

from .base import EnvironmentRuntime
from .fixture_parser import EnvironmentFixture


@dataclass(frozen=True, slots=True)
class FilesystemLimits:
    """Explicit positive integer limits, validated before scratch creation.

    max_entries bounds fixture source entries AND files/directories below the root; max_content_bytes
    is their aggregate file size. max_output_bytes bounds each returned UTF-8 result
    (including list/search JSON); max_snapshot_bytes bounds snapshot JSON separately.
    max_depth counts relative path components; max_path_bytes bounds each UTF-8 path.
    max_seconds is the cooperative lifetime from setup, excluding bounded cleanup.
    """
    max_entries: int
    max_content_bytes: int
    max_output_bytes: int
    max_snapshot_bytes: int
    max_depth: int
    max_path_bytes: int
    max_seconds: int

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field.name} must be a positive integer")
        try:
            seconds = float(self.max_seconds)
        except OverflowError as exc:
            raise ValueError('deadline cannot be represented') from exc
        if not math.isfinite(seconds):
            raise ValueError('deadline must be finite')
        if self.max_depth > 64:
            raise ValueError("descriptor traversal supports at most 64 path components")


class BoundedLocalEnvironmentRuntime(EnvironmentRuntime):
    """Host-owned private scratch with no shell, network, or provider integration.

    Failures may leave partial owned mutations; teardown is explicit and bounded.
    Identity substitution during teardown retains the owned directory for host
    reconciliation rather than deleting an uncertain namespace. Not thread-safe.
    """
    def __init__(self, private_root: str | Path, limits: FilesystemLimits):
        if type(limits) is not FilesystemLimits:
            raise TypeError("exact FilesystemLimits required")
        self._parent_path = os.fspath(private_root)
        self.limits = limits
        self._anchors = []
        self._root_fd = None
        self._name = None
        self._deadline = None
        self._created_identity = None
        self._created = False

    @staticmethod
    def _platform():
        required = (os.open, os.mkdir, os.stat, os.unlink, os.rmdir, os.rename, os.link)
        if (not sys.platform.startswith('linux') or os.name != 'posix' or not hasattr(os, 'O_NOFOLLOW') or
                not hasattr(os, 'O_DIRECTORY') or not hasattr(os, 'O_NONBLOCK') or
                any(fn not in os.supports_dir_fd for fn in required) or
                os.listdir not in os.supports_fd):
            raise RuntimeError("descriptor filesystem facilities unavailable")

    def _parts(self, path, *, root=False):
        if type(path) is not str or not path or '\\' in path or '\x00' in path:
            raise ValueError("invalid relative path")
        if path == '.' and root:
            return ()
        parts = tuple(path.split('/'))
        if (any(p in ('', '.', '..') for p in parts) or path.startswith('/') or
                ':' in parts[0] or len(parts) > self.limits.max_depth or
                len(path.encode('utf-8')) > self.limits.max_path_bytes):
            raise ValueError("invalid relative path")
        return parts

    @staticmethod
    def _identity(value):
        return value.st_dev, value.st_ino

    def _check(self, *, cleanup=False):
        if self._root_fd is None:
            raise RuntimeError("runtime not initialized")
        for fd in (self._anchors[-1][2], self._root_fd):
            private_info = os.fstat(fd)
            if private_info.st_uid != os.getuid() or stat.S_IMODE(private_info.st_mode) != 0o700:
                raise RuntimeError('private directory permissions changed')
        for parent, name, fd in self._anchors:
            if parent is not None and self._identity(os.stat(name, dir_fd=parent,
                    follow_symlinks=False)) != self._identity(os.fstat(fd)):
                raise RuntimeError("private ancestor identity changed")
        if self._identity(os.stat(self._name, dir_fd=self._anchors[-1][2],
                follow_symlinks=False)) != self._identity(os.fstat(self._root_fd)):
            raise RuntimeError("scratch identity changed; cleanup authority uncertain")
        if not cleanup and time.monotonic() >= self._deadline:
            raise TimeoutError("cooperative filesystem deadline exceeded")

    @contextmanager
    def _directory(self, parts, *, create=False, cleanup=False):
        self._check(cleanup=cleanup)
        current = self._root_fd
        opened = []
        try:
            for name in parts:
                if create:
                    try:
                        os.mkdir(name, 0o700, dir_fd=current)
                    except FileExistsError:
                        pass
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=current)
                opened.append((current, name, child))
                current = child
            yield current
            for parent, name, fd in opened:
                if self._identity(os.stat(name, dir_fd=parent, follow_symlinks=False)) != self._identity(os.fstat(fd)):
                    raise RuntimeError("relative ancestor identity changed")
            self._check(cleanup=cleanup)
        finally:
            for _, _, fd in reversed(opened):
                os.close(fd)

    def _inventory(self, *, cleanup=False):
        """Complete bounded inventory, rejecting hostile leaves before using them."""
        rows = []
        identities = {}
        total = 0
        def visit(fd, prefix):
            nonlocal total
            self._check(cleanup=cleanup)
            names = []
            with os.scandir(fd) as iterator:
                for entry in iterator:
                    names.append(entry.name)
                    if len(names) + len(rows) > self.limits.max_entries:
                        raise ValueError("filesystem entry limit exceeded")
            for name in sorted(names):
                parts = prefix + (name,)
                path = '/'.join(parts)
                self._parts(path)
                before = os.stat(name, dir_fd=fd, follow_symlinks=False)
                identities[path] = self._identity(before)
                if stat.S_ISDIR(before.st_mode):
                    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    try:
                        if self._identity(before) != self._identity(os.fstat(child)):
                            raise RuntimeError("directory identity changed")
                        rows.append((path, 'directory', 0))
                        if len(rows) > self.limits.max_entries:
                            raise ValueError("filesystem entry limit exceeded")
                        visit(child, parts)
                        if self._identity(os.stat(name, dir_fd=fd, follow_symlinks=False)) != self._identity(before):
                            raise RuntimeError("directory identity changed")
                    finally:
                        os.close(child)
                elif stat.S_ISREG(before.st_mode) and before.st_nlink == 1:
                    total += before.st_size
                    rows.append((path, 'file', before.st_size))
                    if total > self.limits.max_content_bytes or len(rows) > self.limits.max_entries:
                        raise ValueError("filesystem content/entry limit exceeded")
                else:
                    raise ValueError("symlink, hardlink or special entry rejected")
        visit(self._root_fd, ())
        self._inventory_identities = identities
        return rows

    def _capacity(self, files, directories, *, replacing=None):
        current = {p: (kind, size) for p, kind, size in self._inventory()}
        if replacing is not None:
            current = {p: row for p, row in current.items() if p != replacing and not p.startswith(replacing + '/')}
        for path in directories:
            existing = current.get(path)
            if existing and existing[0] != 'directory':
                raise FileExistsError("directory collides with file")
            current[path] = ('directory', 0)
        for path, content in files.items():
            existing = current.get(path)
            if existing and existing[0] != 'file':
                raise IsADirectoryError(path)
            current[path] = ('file', len(content))
        if len(current) > self.limits.max_entries or sum(size for _, size in current.values()) > self.limits.max_content_bytes:
            raise ValueError("filesystem capacity exceeded")

    def _parents(self, parts):
        return ['/'.join(parts[:index]) for index in range(1, len(parts))]

    def setup(self, fixture):
        self._platform()
        if self._root_fd is not None or self._anchors:
            raise RuntimeError("runtime already initialized or cleanup unresolved")
        if type(fixture) is not EnvironmentFixture or type(fixture.directories) is not list or type(fixture.files) is not dict:
            raise TypeError("exact fixture required")
        if len(fixture.directories) + len(fixture.files) > self.limits.max_entries:
            raise ValueError('fixture source entry limit exceeded')
        directories = set()
        files = {}
        fixture_bytes = 0
        for path in fixture.directories:
            parts = self._parts(path, root=True)
            directories.update(self._parents(parts) + (['/'.join(parts)] if parts else []))
            if len(directories) > self.limits.max_entries:
                raise ValueError('fixture entry limit exceeded')
        for path, content in fixture.files.items():
            parts = self._parts(path)
            if type(content) is not str:
                raise TypeError("fixture content must be text")
            if len(content) > self.limits.max_content_bytes:
                raise ValueError('fixture content limit exceeded')
            raw = content.encode('utf-8')
            fixture_bytes += len(raw)
            if fixture_bytes > self.limits.max_content_bytes:
                raise ValueError('fixture content limit exceeded')
            files[path] = raw
            directories.update(self._parents(parts))
            if len(files) + len(directories) > self.limits.max_entries:
                raise ValueError('fixture entry limit exceeded')
        if (set(files) & directories or len(files) + len(directories) > self.limits.max_entries or
                sum(map(len, files.values())) > self.limits.max_content_bytes):
            raise ValueError("invalid or oversized fixture")
        parent = Path(self._parent_path)
        if not parent.is_absolute() or '..' in parent.parts:
            raise ValueError("private parent must be absolute")
        try:
            fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            self._anchors.append((None, None, fd))
            for name in parent.parts[1:]:
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                self._anchors.append((fd, name, child))
                fd = child
            info = os.fstat(fd)
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise ValueError("private parent must be owned mode 0700")
            self._name = 'fixture-' + secrets.token_hex(16)
            os.mkdir(self._name, 0o700, dir_fd=fd)
            self._created = True
            self._created_identity = self._identity(os.stat(self._name, dir_fd=fd, follow_symlinks=False))
            opened_root = os.open(self._name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            try:
                opened_info = os.fstat(opened_root)
                if self._identity(opened_info) != self._created_identity:
                    raise RuntimeError('created scratch identity changed before opening')
                if opened_info.st_uid != os.getuid() or stat.S_IMODE(opened_info.st_mode) != 0o700:
                    raise RuntimeError('created scratch permissions changed before opening')
            except BaseException:
                os.close(opened_root)
                raise
            self._root_fd = opened_root
            self._deadline = time.monotonic() + self.limits.max_seconds
            for path in sorted(directories, key=lambda p: (p.count('/'), p)):
                self.mkdir(path)
            for path, content in fixture.files.items():
                self.write_text(path, content)
            self._check()
        except BaseException:
            if self._root_fd is not None:
                self.teardown()
            elif self._created_identity is not None:
                # Opening may fail after mkdir: do not forget created ownership.
                parent_fd = self._anchors[-1][2]
                if self._identity(os.stat(self._name, dir_fd=parent_fd, follow_symlinks=False)) != self._created_identity:
                    raise RuntimeError('unopened scratch identity changed; cleanup uncertain')
                os.rmdir(self._name, dir_fd=parent_fd)
                self._close()
            elif self._created:
                raise RuntimeError('created scratch identity unestablished; cleanup uncertain') from None
            else:
                self._close()
            raise

    def _close(self):
        if self._root_fd is not None:
            os.close(self._root_fd)
        for _, _, fd in reversed(self._anchors):
            os.close(fd)
        self._root_fd = None
        self._anchors = []
        self._name = None
        self._deadline = None
        self._created_identity = None
        self._created = False

    def teardown(self):
        if self._root_fd is None:
            if self._created and self._created_identity is None:
                raise RuntimeError('created scratch identity unestablished; cleanup uncertain')
            if self._created_identity is not None:
                parent_fd = self._anchors[-1][2]
                if self._identity(os.stat(self._name, dir_fd=parent_fd, follow_symlinks=False)) != self._created_identity:
                    raise RuntimeError('unopened scratch identity changed; cleanup uncertain')
                os.rmdir(self._name, dir_fd=parent_fd)
                self._close()
            return
        # Cleanup deliberately ignores the expired lifetime, but never its work bounds.
        self._check(cleanup=True)
        rows = self._inventory(cleanup=True)
        identities = dict(self._inventory_identities)
        for path, kind, _ in sorted(rows, key=lambda r: (r[0].count('/'), r[0]), reverse=True):
            parts = self._parts(path)
            with self._directory(parts[:-1], cleanup=True) as fd:
                if self._identity(os.stat(parts[-1], dir_fd=fd, follow_symlinks=False)) != identities[path]:
                    raise RuntimeError('cleanup entry identity changed; authority uncertain')
                (os.rmdir if kind == 'directory' else os.unlink)(parts[-1], dir_fd=fd)
        self._check(cleanup=True)
        os.rmdir(self._name, dir_fd=self._anchors[-1][2])
        self._close()

    def mkdir(self, path):
        parts = self._parts(path, root=True)
        self._capacity({}, self._parents(parts) + (['/'.join(parts)] if parts else []))
        with self._directory(parts, create=True):
            pass

    def write_text(self, path, content):
        parts = self._parts(path)
        if type(content) is not str:
            raise TypeError("content must be text")
        if len(content) > self.limits.max_content_bytes:
            raise ValueError('filesystem content limit exceeded')
        raw = content.encode('utf-8')
        self._capacity({path: raw}, self._parents(parts))
        with self._directory(parts[:-1], create=True) as parent:
            flags = os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            try:
                fd = os.open(parts[-1], flags | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent)
            except FileExistsError:
                fd = os.open(parts[-1], flags, dir_fd=parent)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError("non-private regular file rejected")
                os.ftruncate(fd, 0)
                offset = 0
                while offset < len(raw):
                    self._check()
                    written = os.write(fd, raw[offset:])
                    if written <= 0:
                        raise OSError("write made no progress")
                    offset += written
                os.fsync(fd)
                if self._identity(os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)) != self._identity(info):
                    raise RuntimeError("file identity changed")
            finally:
                os.close(fd)

    def _read(self, path):
        parts = self._parts(path)
        with self._directory(parts[:-1]) as parent:
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            try:
                before = os.fstat(fd)
                if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > self.limits.max_content_bytes:
                    raise ValueError("unsafe or oversized file")
                raw = bytearray()
                while True:
                    self._check()
                    chunk = os.read(fd, min(65536, self.limits.max_content_bytes + 1 - len(raw)))
                    if not chunk:
                        break
                    raw.extend(chunk)
                    if len(raw) > self.limits.max_content_bytes:
                        raise ValueError("file content limit exceeded")
                after = os.fstat(fd)
                if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) or self._identity(os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)) != self._identity(before):
                    raise RuntimeError("file changed during read")
                return bytes(raw)
            finally:
                os.close(fd)

    def _output(self, result, *, snapshot=False):
        raw = result.encode('utf-8') if type(result) is str else json.dumps(result, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        if len(raw) > (self.limits.max_snapshot_bytes if snapshot else self.limits.max_output_bytes):
            raise ValueError("returned output byte limit exceeded")
        return result

    def read_text(self, path):
        self._inventory()
        return self._output(self._read(path).decode('utf-8'))

    def list_dir(self, path='.'):
        parts = self._parts(path, root=True)
        self._inventory()
        with self._directory(parts) as fd:
            names = []
            output_bytes = 2
            with os.scandir(fd) as iterator:
                for entry in iterator:
                    output_bytes += len(json.dumps(entry.name, ensure_ascii=False).encode('utf-8')) + bool(names)
                    if output_bytes > self.limits.max_output_bytes:
                        raise ValueError('returned output byte limit exceeded')
                    names.append(entry.name)
                    if len(names) > self.limits.max_entries:
                        raise ValueError('directory entry limit exceeded')
            return self._output(sorted(names))

    def exists(self, path):
        parts = self._parts(path, root=True)
        inventory = {p for p, _, _ in self._inventory()}
        return not parts or path in inventory

    def delete(self, path, recursive=False):
        self._parts(path)
        rows = self._inventory()
        identities = dict(self._inventory_identities)
        selected = [r for r in rows if r[0] == path or r[0].startswith(path + '/')]
        if not selected:
            return
        if not recursive and len(selected) > 1:
            raise OSError("directory not empty")
        for current, kind, _ in sorted(selected, key=lambda r: (r[0].count('/'), r[0]), reverse=True):
            components = self._parts(current)
            with self._directory(components[:-1]) as parent:
                if self._identity(os.stat(components[-1], dir_fd=parent, follow_symlinks=False)) != identities[current]:
                    raise RuntimeError('delete entry identity changed')
                (os.rmdir if kind == 'directory' else os.unlink)(components[-1], dir_fd=parent)

    def copy(self, path, new_path, overwrite=False):
        self._parts(path)
        destination = self._parts(new_path)
        if new_path == path or new_path.startswith(path + '/') or path.startswith(new_path + '/'):
            raise ValueError("overlapping copy paths")
        rows = self._inventory()
        selected = [r for r in rows if r[0] == path or r[0].startswith(path + '/')]
        if not selected:
            raise FileNotFoundError(path)
        replacing = self.exists(new_path)
        if replacing and not overwrite:
            raise FileExistsError(new_path)
        files = {}
        directories = self._parents(destination)
        for current, kind, _ in selected:
            target = new_path + current[len(path):]
            self._parts(target)
            if kind == 'directory':
                directories.append(target)
            else:
                files[target] = self._read(current)
        self._capacity(files, directories, replacing=new_path if replacing else None)
        if replacing:
            # Replacement is bounded, but is not transactional or crash atomic.
            self.delete(new_path, recursive=True)
        for directory in sorted(set(directories), key=lambda p: (p.count('/'), p)):
            self.mkdir(directory)
        for target, raw in files.items():
            self.write_text(target, raw.decode('utf-8'))

    def move(self, path, new_path, overwrite=False):
        # Requires temporary capacity for source AND copy, even if final size fits.
        # Reuses bounded copy/delete; no atomic rename/transaction promise.
        self.copy(path, new_path, overwrite=overwrite)
        self.delete(path, recursive=True)

    def search(self, query, path='.'):
        if type(query) is not str or len(query.encode('utf-8')) > self.limits.max_output_bytes:
            raise ValueError("invalid search query")
        self._parts(path, root=True)
        rows = self._inventory()
        if path != '.' and not any(p == path for p, _, _ in rows):
            raise FileNotFoundError(path)
        result = []
        read_bytes = 0
        for current, kind, _ in rows:
            if kind == 'file' and (path == '.' or current == path or current.startswith(path + '/')):
                if query.lower() in current.lower():
                    result.append(current)
                    continue
                raw = self._read(current)
                read_bytes += len(raw)
                if read_bytes > self.limits.max_content_bytes:
                    raise ValueError('search aggregate content limit exceeded')
                if query.lower() in raw.decode('utf-8').lower():
                    result.append(current)
        return self._output(result)

    def snapshot(self, limit=200):
        if type(limit) is not int or limit <= 0 or limit > self.limits.max_entries:
            raise ValueError("invalid snapshot entry limit")
        rows = self._inventory()
        selected = rows[:limit]
        return self._output({'runtime': 'bounded_local',
            'directories': [p for p, kind, _ in selected if kind == 'directory'],
            'files': [{'path': p, 'size': size} for p, kind, size in selected if kind == 'file'],
            'truncated': len(rows) > limit}, snapshot=True)
