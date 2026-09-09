"""Descriptor-relative I/O for hostile shared Modal mounts.

The locked Linux runtime requires the secure dir_fd implementation. Platforms
without it use best-effort pathname checks, not equivalent race protection.
"""
from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path


_BINARY = getattr(os, "O_BINARY", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_SECURE_DIRFD = (
    os.name == "posix"
    and _NOFOLLOW != 0
    and _DIRECTORY != 0
    and os.open in os.supports_dir_fd
    and os.mkdir in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
    and os.stat in os.supports_follow_symlinks
)


def _is_link_or_reparse(path: Path, info: os.stat_result) -> bool:
    return path.is_symlink() or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _relative(root: Path, path: Path) -> Path:
    try:
        value = path.absolute().relative_to(root.absolute())
    except ValueError:
        raise ValueError("runtime path escapes its mounted root") from None
    if not value.parts or any(part in {"", ".", ".."} for part in value.parts):
        raise ValueError("runtime path is invalid")
    return value


def _checked_parent(root: Path, relative: Path, *, create: bool) -> None:
    if create:
        root.mkdir(parents=True, exist_ok=True)
    for candidate in (root, *(root.joinpath(*relative.parts[:index]) for index in range(1, len(relative.parts)))):
        if create:
            candidate.mkdir(exist_ok=True)
        try:
            info = candidate.lstat()
        except OSError:
            raise ValueError("runtime path parent is unavailable") from None
        if not stat.S_ISDIR(info.st_mode) or _is_link_or_reparse(candidate, info):
            raise ValueError("runtime path parent is not a trusted directory")


def _parent_handle(
    root: Path, path: Path, *, create: bool
) -> tuple[int | None, str, Path]:
    relative = _relative(root, path)
    if not _SECURE_DIRFD:
        _checked_parent(root, relative, create=create)
        return None, relative.parts[-1], path
    if create:
        root.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC
    try:
        descriptor = os.open(root, flags)
    except OSError:
        raise ValueError("runtime mounted root is unavailable") from None
    try:
        for component in relative.parts[:-1]:
            if create:
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except Exception:
        os.close(descriptor)
        raise ValueError("runtime path parent is not a trusted directory") from None
    return descriptor, relative.parts[-1], path


def _leaf_info(parent: int | None, leaf: str, path: Path) -> os.stat_result:
    if parent is None:
        return path.lstat()
    return os.stat(leaf, dir_fd=parent, follow_symlinks=False)


def _open_leaf(
    parent: int | None,
    leaf: str,
    path: Path,
    flags: int,
    mode: int | None = None,
) -> int:
    if parent is None:
        return os.open(path, flags) if mode is None else os.open(path, flags, mode)
    return (
        os.open(leaf, flags, dir_fd=parent)
        if mode is None
        else os.open(leaf, flags, mode, dir_fd=parent)
    )


def read_regular(root: Path, path: Path, maximum: int) -> bytes:
    """Read one bounded regular leaf without reopening an ancestor path."""
    if type(maximum) is not int or maximum < 0:
        raise ValueError("mounted member bound must be a nonnegative exact integer")
    parent, leaf, full_path = _parent_handle(root, path, create=False)
    try:
        before = _leaf_info(parent, leaf, full_path)
        if (
            not stat.S_ISREG(before.st_mode)
            or (parent is None and _is_link_or_reparse(full_path, before))
            or before.st_size > maximum
        ):
            raise ValueError("mounted member is not a bounded regular file")
        descriptor = _open_leaf(
            parent, leaf, full_path,
            os.O_RDONLY | _BINARY | _NOFOLLOW | _CLOEXEC,
        )
    except Exception:
        raise ValueError("mounted member unavailable") from None
    finally:
        if parent is not None:
            os.close(parent)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev, opened.st_ino
        ) != (before.st_dev, before.st_ino):
            raise ValueError("mounted member changed before read")
        content = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    if len(content) > maximum or len(content) != before.st_size or (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ) != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
        raise ValueError("mounted member changed during read")
    return content


def hash_regular(root: Path, path: Path, maximum: int) -> tuple[int, str]:
    """Hash a bounded regular leaf with at most one MiB of working buffer.

    As with read_regular, the Linux path is opened relative to a retained
    parent descriptor; a pathname replacement cannot redirect an ongoing read.
    The returned size/digest describes that opened file, not a future read.
    """
    if type(maximum) is not int or maximum < 0:
        raise ValueError("mounted member bound must be a nonnegative exact integer")
    parent, leaf, full_path = _parent_handle(root, path, create=False)
    try:
        before = _leaf_info(parent, leaf, full_path)
        if (
            not stat.S_ISREG(before.st_mode)
            or (parent is None and _is_link_or_reparse(full_path, before))
            or not 0 <= before.st_size <= maximum
        ):
            raise ValueError("mounted member is not a bounded regular file")
        descriptor = _open_leaf(
            parent, leaf, full_path,
            os.O_RDONLY | _BINARY | _NOFOLLOW | _CLOEXEC,
        )
    except Exception:
        raise ValueError("mounted member unavailable") from None
    finally:
        if parent is not None:
            os.close(parent)
    digest = hashlib.sha256()
    size = 0
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev, opened.st_ino
        ) != (before.st_dev, before.st_ino):
            raise ValueError("mounted member changed before hash")
        while True:
            chunk = stream.read(min(1024 * 1024, maximum + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > maximum:
                raise ValueError("mounted member exceeds its bound")
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    if size != before.st_size or (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ) != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
        raise ValueError("mounted member changed during hash")
    return size, digest.hexdigest()


def write_exclusive(root: Path, path: Path, content: bytes) -> None:
    """Create one leaf relative to a retained trusted parent descriptor."""
    if type(content) is not bytes:
        raise TypeError("mounted write requires exact bytes")
    parent, leaf, full_path = _parent_handle(root, path, create=True)
    try:
        descriptor = _open_leaf(
            parent,
            leaf,
            full_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _BINARY | _NOFOLLOW | _CLOEXEC,
            0o600,
        )
    finally:
        if parent is not None:
            os.close(parent)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def claim_directory(root: Path, path: Path) -> None:
    """Exclusively create one output directory; existing paths are collisions."""
    parent, leaf, full_path = _parent_handle(root, path, create=True)
    try:
        if parent is None:
            full_path.mkdir(mode=0o700)
        else:
            os.mkdir(leaf, 0o700, dir_fd=parent)
    finally:
        if parent is not None:
            os.close(parent)


def list_regular_sizes(
    root: Path, directory: Path, maximum_entries: int,
) -> tuple[tuple[str, int], ...]:
    """Bounded regular-file inventory of one retained output directory.

    This is a snapshot, not a guarantee against later mutation. Artifact
    readers must independently revalidate their exact inventory and content.
    """
    if type(maximum_entries) is not int or maximum_entries < 0:
        raise ValueError("directory entry bound must be a nonnegative exact integer")
    # Only the parent is opened; this synthetic leaf is never read or created.
    parent, _, _ = _parent_handle(root, directory / ".inventory", create=False)
    result: list[tuple[str, int]] = []
    try:
        with os.scandir(directory if parent is None else parent) as entries:
            for entry in entries:
                if len(result) >= maximum_entries:
                    raise ValueError("output directory exceeds its entry bound")
                info = entry.stat(follow_symlinks=False)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or (parent is None and _is_link_or_reparse(directory / entry.name, info))
                ):
                    raise ValueError("output directory contains a nonregular member")
                result.append((entry.name, info.st_size))
    finally:
        if parent is not None:
            os.close(parent)
    return tuple(sorted(result))


def copy_regular(
    source_root: Path,
    source: Path,
    destination_root: Path,
    destination: Path,
    *,
    maximum: int,
) -> tuple[int, str]:
    """Copy one bounded regular leaf between retained mount directories."""
    if type(maximum) is not int or maximum < 0:
        raise ValueError("mounted member bound must be a nonnegative exact integer")
    source_parent, source_leaf, source_path = _parent_handle(
        source_root, source, create=False
    )
    try:
        destination_parent, destination_leaf, destination_path = _parent_handle(
            destination_root, destination, create=True
        )
    except Exception:
        if source_parent is not None:
            os.close(source_parent)
        raise
    source_descriptor: int | None = None
    try:
        before = _leaf_info(source_parent, source_leaf, source_path)
        if (
            not stat.S_ISREG(before.st_mode)
            or (source_parent is None and _is_link_or_reparse(source_path, before))
            or not 0 <= before.st_size <= maximum
        ):
            raise ValueError("runtime artifact is not a bounded regular file")
        source_descriptor = _open_leaf(
            source_parent,
            source_leaf,
            source_path,
            os.O_RDONLY | _BINARY | _NOFOLLOW | _CLOEXEC,
        )
        destination_descriptor = _open_leaf(
            destination_parent,
            destination_leaf,
            destination_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _BINARY | _NOFOLLOW | _CLOEXEC,
            0o600,
        )
    except Exception:
        if source_descriptor is not None:
            os.close(source_descriptor)
        raise
    finally:
        if source_parent is not None:
            os.close(source_parent)
        if destination_parent is not None:
            os.close(destination_parent)
    digest = hashlib.sha256()
    size = 0
    with os.fdopen(source_descriptor, "rb") as source_stream, os.fdopen(
        destination_descriptor, "wb"
    ) as target:
        opened = os.fstat(source_stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev, opened.st_ino
        ) != (before.st_dev, before.st_ino):
            raise ValueError("runtime artifact changed before publication")
        for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
            size += len(chunk)
            if size > maximum:
                raise ValueError("runtime artifact exceeds its bound")
            digest.update(chunk)
            target.write(chunk)
        target.flush()
        os.fsync(target.fileno())
        after = os.fstat(source_stream.fileno())
    if size != before.st_size or (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ) != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
        raise ValueError("runtime artifact changed during publication")
    return size, digest.hexdigest()


__all__ = [
    "claim_directory", "copy_regular", "hash_regular", "list_regular_sizes",
    "read_regular", "write_exclusive",
]
