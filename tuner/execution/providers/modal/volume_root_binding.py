"""Linux-only, retained-descriptor transport for a provider-presented root link.

The v2 Modal worker authenticates the signed dispatch, exact Volume identities,
and marker digests before constructing a binding. Marker equality demonstrates
that this process sees the bytes placed in a Volume; it is not provider
attestation of a Volume ID.

No pathname operation inside the bound root follows a symlink. Outside that
root, traversed directories must be owned by root or this process's UID and
have no group/other write bit; private scratch under sticky `/tmp` additionally
requires an owner-owned 0700 anchor. POSIX ACLs and another actor with this UID
remain outside this module's trust model; the caller must exclude them.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import sys
from dataclasses import dataclass


_DIR = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_READ = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_WRITE = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_MAX_COMPONENTS = 32
_MAX_PATH = 2048
_MAX_NAME = 255
_MAX_COPY_BYTES = 32 * 1024 * 1024 * 1024
_COPY_CHUNK = 1024 * 1024
_PROVIDER_VOLUME_ROOT = "/__modal/volumes"
_VOLUME_ID = re.compile(r"vo-[A-Za-z0-9]+\Z")
PUBLICATION_DIAGNOSTICS = frozenset({
    "SOURCE_CHAIN_ROOT", "SOURCE_CHAIN_TMP", "SOURCE_CHAIN_TMP_OWNER",
    "SOURCE_CHAIN_TMP_MODE_NONWRITABLE", "SOURCE_CHAIN_TMP_MODE_WRITABLE",
    "SOURCE_CHAIN_OWNER",
    "SOURCE_CHAIN_MODE", "SOURCE_CHAIN_OPEN",
    "CLAIM_ROOT_ADMISSION", "CLAIM_PARENT", "CLAIM_CREATE_EXISTS", "CLAIM_CREATE_DENIED",
    "CLAIM_CREATE_OS", "CLAIM_IDENTITY", "CLAIM_RECHECK",
    "COPY_ROOT_ADMISSION", "COPY_MEMBER_PARENT", "COPY_SOURCE_ADMISSION",
    "COPY_SOURCE_OPEN", "COPY_STREAM",
    "COPY_DEST_CREATE_EXISTS", "COPY_DEST_CREATE_DENIED", "COPY_DEST_CREATE_OS",
    "COPY_DEST_IDENTITY", "COPY_STREAM_READ", "COPY_STREAM_WRITE",
    "COPY_STREAM_HASH", "COPY_STREAM_FSYNC", "COPY_SOURCE_RECHECK",
    "COPY_DEST_RECHECK", "COPY_MEMBER_RECHECK", "COPY_PRIVATE_RECHECK",
    "COPY_ROOT_RECHECK",
})


class VolumeRootBindingError(ValueError):
    """Closed error for an unavailable or changed bound root."""

    def __init__(self, code: str | None = None):
        self.code = code if code in PUBLICATION_DIAGNOSTICS else None
        super().__init__("volume root binding invalid")


def _invalid(code: str | None = None) -> VolumeRootBindingError:
    return VolumeRootBindingError(code)


def _create_code(prefix: str, error: BaseException) -> str:
    if isinstance(error, FileExistsError):
        return prefix + "_EXISTS"
    if isinstance(error, PermissionError):
        return prefix + "_DENIED"
    return prefix + "_OS"


def _close_all(descriptors: tuple[int, ...] | list[int]) -> None:
    failed = False
    for descriptor in reversed(descriptors):
        try:
            os.close(descriptor)
        except OSError:
            failed = True
    if failed:
        raise _invalid()


def _linux() -> None:
    if (
        sys.platform != "linux"
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_PATH")
        or os.open not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
        or os.stat not in os.supports_follow_symlinks
        or os.readlink not in os.supports_dir_fd
    ):
        raise _invalid()


def _parts(path: str, *, absolute: bool) -> tuple[str, ...]:
    if type(path) is not str or not path or len(path) > _MAX_PATH or path.startswith("/") != absolute:
        raise _invalid()
    pieces = path.split("/")
    if absolute:
        pieces = pieces[1:]
    if not pieces or len(pieces) > _MAX_COMPONENTS or any(
        not piece or piece in (".", "..") or len(piece) > _MAX_NAME or "\x00" in piece
        for piece in pieces
    ):
        raise _invalid()
    return tuple(pieces)


def _identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid)


def _trusted_dir(info: os.stat_result) -> None:
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid not in (0, os.geteuid())
        or info.st_mode & 0o022
    ):
        raise _invalid()


def _open_chain(parts: tuple[str, ...]) -> list[int]:
    descriptors = [os.open("/", _DIR)]
    try:
        _trusted_dir(os.fstat(descriptors[0]))
        for part in parts:
            next_fd = os.open(part, _DIR, dir_fd=descriptors[-1])
            descriptors.append(next_fd)
            _trusted_dir(os.fstat(next_fd))
        return descriptors
    except BaseException:
        try:
            _close_all(descriptors)
        except VolumeRootBindingError:
            pass
        raise


@dataclass(frozen=True)
class _Chain:
    parts: tuple[str, ...]
    descriptors: tuple[int, ...]
    identities: tuple[tuple[int, int, int, int], ...]


def _chain(parts: tuple[str, ...]) -> _Chain:
    descriptors = _open_chain(parts)
    try:
        identities = tuple(_identity(os.fstat(fd)) for fd in descriptors)
        return _Chain(parts, tuple(descriptors), identities)
    except BaseException:
        try:
            _close_all(descriptors)
        except VolumeRootBindingError:
            pass
        raise


def _private_chain(parts: tuple[str, ...]) -> _Chain:
    """Retain a private scratch parent, admitting the standard sticky /tmp only."""
    try:
        descriptors = [os.open("/", _DIR)]
    except OSError:
        raise _invalid("SOURCE_CHAIN_OPEN") from None
    try:
        try:
            _trusted_dir(os.fstat(descriptors[0]))
        except (OSError, VolumeRootBindingError):
            raise _invalid("SOURCE_CHAIN_ROOT") from None
        in_tmp = False
        for index, part in enumerate(parts):
            try:
                fd = os.open(part, _DIR, dir_fd=descriptors[-1])
            except OSError:
                raise _invalid("SOURCE_CHAIN_OPEN") from None
            descriptors.append(fd)
            info = os.fstat(fd)
            if index == 0 and part == "tmp":
                if not stat.S_ISDIR(info.st_mode):
                    raise _invalid("SOURCE_CHAIN_TMP")
                if info.st_uid != 0:
                    raise _invalid("SOURCE_CHAIN_TMP_OWNER")
                mode = stat.S_IMODE(info.st_mode)
                if mode != 0o1777:
                    raise _invalid("SOURCE_CHAIN_TMP_MODE_WRITABLE" if mode & 0o022
                                   else "SOURCE_CHAIN_TMP_MODE_NONWRITABLE")
                in_tmp = True
            elif in_tmp:
                mode = stat.S_IMODE(info.st_mode)
                # The first child of sticky /tmp must be the private 0700
                # anchor. SDK-created descendants may be 0755 under it, but
                # must remain owner-owned and not writable by anyone else.
                if not stat.S_ISDIR(info.st_mode):
                    raise _invalid("SOURCE_CHAIN_MODE")
                if info.st_uid != os.geteuid():
                    raise _invalid("SOURCE_CHAIN_OWNER")
                if (index == 1 and mode != 0o700) or (index > 1 and mode & 0o022):
                    raise _invalid("SOURCE_CHAIN_MODE")
            else:
                if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o022:
                    raise _invalid("SOURCE_CHAIN_MODE")
                if info.st_uid not in (0, os.geteuid()):
                    raise _invalid("SOURCE_CHAIN_OWNER")
        final = os.fstat(descriptors[-1])
        if not parts or final.st_uid != os.geteuid():
            raise _invalid("SOURCE_CHAIN_OWNER")
        if not in_tmp and stat.S_IMODE(final.st_mode) != 0o700:
            raise _invalid("SOURCE_CHAIN_MODE")
        return _Chain(parts, tuple(descriptors), tuple(_identity(os.fstat(fd)) for fd in descriptors))
    except BaseException:
        try:
            _close_all(descriptors)
        except VolumeRootBindingError:
            pass
        raise


def _recheck_chain(chain: _Chain) -> None:
    if _identity(os.fstat(chain.descriptors[0])) != chain.identities[0]:
        raise _invalid()
    for index, part in enumerate(chain.parts, start=1):
        fd = os.open(part, _DIR, dir_fd=chain.descriptors[index - 1])
        try:
            if _identity(os.fstat(fd)) != chain.identities[index]:
                raise _invalid()
        finally:
            os.close(fd)


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _copy_bounds(expected_size: int, expected_sha256: str, maximum: int) -> None:
    if (
        type(expected_size) is not int or type(maximum) is not int
        or not 0 <= expected_size <= maximum <= _MAX_COPY_BYTES
        or type(expected_sha256) is not str or len(expected_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in expected_sha256)
    ):
        raise _invalid()


def _open_exact_source(parent: int, leaf: str, expected_size: int) -> tuple[int, tuple[int, int, int, int, int, int]]:
    before = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != expected_size:
        raise _invalid("COPY_SOURCE_ADMISSION")
    try:
        fd = os.open(leaf, _READ, dir_fd=parent)
    except OSError:
        raise _invalid("COPY_SOURCE_OPEN") from None
    try:
        if _file_identity(os.fstat(fd)) != _file_identity(before):
            raise _invalid("COPY_SOURCE_ADMISSION")
        return fd, _file_identity(before)
    except BaseException:
        os.close(fd)
        raise


def _stream_exact(source: int, destination: int, expected_size: int, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    count = 0
    while True:
        try:
            chunk = os.read(source, min(_COPY_CHUNK, expected_size - count + 1))
        except OSError:
            raise _invalid("COPY_STREAM_READ") from None
        if not chunk:
            break
        count += len(chunk)
        if count > expected_size:
            raise _invalid("COPY_STREAM_HASH")
        digest.update(chunk)
        view = memoryview(chunk)
        while view:
            try:
                written = os.write(destination, view)
            except OSError:
                raise _invalid("COPY_STREAM_WRITE") from None
            if written <= 0:
                raise _invalid("COPY_STREAM_WRITE")
            view = view[written:]
    if count != expected_size or digest.hexdigest() != expected_sha256:
        raise _invalid("COPY_STREAM_HASH")
    try:
        os.fsync(destination)
    except OSError:
        raise _invalid("COPY_STREAM_FSYNC") from None


def _remove_owned_leaf(parent: int, leaf: str, identity: tuple[int, int, int, int]) -> None:
    """Best-effort cleanup only when the name still denotes our exclusive file."""
    try:
        current = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        if stat.S_ISREG(current.st_mode) and current.st_nlink == 1 and _identity(current) == identity:
            os.unlink(leaf, dir_fd=parent)
    except OSError:
        pass


class VolumeRootBinding:
    """One verified root with bounded descriptor-relative file operations.

    Construct with ``bind`` and close explicitly or use as a context manager.
    The supplied marker digest must come from an authenticated, fresh claim.
    The target must be exactly ``/__modal/volumes/<volume_id>``. The Volume ID
    and marker name/digest must come from an authenticated fresh claim; this
    module cannot establish their provider provenance. A caller binding several
    roots must reject duplicate IDs, marker names, and marker digests.
    """

    def __init__(self) -> None:
        raise TypeError("use bind")

    @classmethod
    def bind(
        cls, *, root_path: str, volume_id: str,
        marker_name: str, marker_sha256: str,
    ) -> "VolumeRootBinding":
        _linux()
        root_parts = _parts(root_path, absolute=True)
        if (
            len(root_parts) < 2
            or type(volume_id) is not str
            or len(volume_id) > 256
            or _VOLUME_ID.fullmatch(volume_id) is None
            or type(marker_name) is not str
            or _parts(marker_name, absolute=False) != (marker_name,)
            or type(marker_sha256) is not str
            or len(marker_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in marker_sha256)
        ):
            raise _invalid()
        expected_target = _PROVIDER_VOLUME_ROOT + "/" + volume_id
        expected_parts = _parts(expected_target, absolute=True)
        root_parent = None
        target = None
        link_fd = None
        try:
            root_parent = _chain(root_parts[:-1])
            name = root_parts[-1]
            link_info = os.stat(name, dir_fd=root_parent.descriptors[-1], follow_symlinks=False)
            if not stat.S_ISLNK(link_info.st_mode):
                raise _invalid()
            link_fd = os.open(
                name, os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=root_parent.descriptors[-1],
            )
            if _file_identity(os.fstat(link_fd)) != _file_identity(link_info):
                raise _invalid()
            link_text = os.readlink(name, dir_fd=root_parent.descriptors[-1])
            target_parts = _parts(link_text, absolute=True)
            if target_parts != expected_parts:
                raise _invalid()
            target = _chain(target_parts)
            instance = object.__new__(cls)
            instance._root_parent = root_parent
            instance._target = target
            instance._link_fd = link_fd
            instance._root_name = name
            instance._link_identity = _file_identity(link_info)
            instance._link_text = link_text
            instance._volume_id = volume_id
            instance._marker_name = marker_name
            instance._marker_sha256 = marker_sha256
            instance._closed = False
            instance._check_root()
            marker = instance.read_regular(marker_name, 32)
            if len(marker) != 32 or hashlib.sha256(marker).hexdigest() != marker_sha256:
                raise _invalid()
            instance._check_root()
            return instance
        except BaseException as error:
            if link_fd is not None:
                try:
                    os.close(link_fd)
                except OSError:
                    pass
            if target is not None:
                try:
                    _close_all(target.descriptors)
                except VolumeRootBindingError:
                    pass
            if root_parent is not None:
                try:
                    _close_all(root_parent.descriptors)
                except VolumeRootBindingError:
                    pass
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise _invalid() from None

    @property
    def volume_id(self) -> str:
        return self._volume_id

    @property
    def marker_name(self) -> str:
        return self._marker_name

    @property
    def marker_sha256(self) -> str:
        return self._marker_sha256

    def _check_root(self) -> None:
        if self._closed:
            raise _invalid()
        _recheck_chain(self._root_parent)
        info = os.stat(self._root_name, dir_fd=self._root_parent.descriptors[-1], follow_symlinks=False)
        if _file_identity(os.fstat(self._link_fd)) != self._link_identity or _file_identity(info) != self._link_identity or os.readlink(
            self._root_name, dir_fd=self._root_parent.descriptors[-1]
        ) != self._link_text:
            raise _invalid()
        _recheck_chain(self._target)

    def _member_parent(self, path: str) -> tuple[int, str, list[int]]:
        pieces = _parts(path, absolute=False)
        descriptors: list[int] = []
        parent = self._target.descriptors[-1]
        try:
            for part in pieces[:-1]:
                parent = os.open(part, _DIR, dir_fd=parent)
                descriptors.append(parent)
            return parent, pieces[-1], descriptors
        except BaseException:
            for fd in reversed(descriptors):
                os.close(fd)
            raise

    def _check_member_parents(self, path: str, descriptors: list[int]) -> None:
        pieces = _parts(path, absolute=False)
        if len(pieces) - 1 != len(descriptors):
            raise _invalid()
        parent = self._target.descriptors[-1]
        for part, held in zip(pieces[:-1], descriptors, strict=True):
            current = os.open(part, _DIR, dir_fd=parent)
            try:
                if _identity(os.fstat(current)) != _identity(os.fstat(held)):
                    raise _invalid()
            finally:
                os.close(current)
            parent = held

    def read_regular(self, path: str, maximum: int) -> bytes:
        if type(maximum) is not int or not 0 <= maximum <= 1_073_741_824:
            raise _invalid()
        self._check_root()
        parents: list[int] = []
        leaf_fd = None
        try:
            parent, leaf, parents = self._member_parent(path)
            before = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > maximum:
                raise _invalid()
            leaf_fd = os.open(leaf, _READ, dir_fd=parent)
            opened = os.fstat(leaf_fd)
            if not stat.S_ISREG(opened.st_mode) or _file_identity(opened) != _file_identity(before):
                raise _invalid()
            data = bytearray()
            while len(data) <= maximum:
                chunk = os.read(leaf_fd, min(1_048_576, maximum + 1 - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
            if len(data) != before.st_size or _file_identity(os.fstat(leaf_fd)) != _file_identity(before):
                raise _invalid()
            self._check_member_parents(path, parents)
            self._check_root()
            return bytes(data)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise _invalid() from None
        finally:
            if leaf_fd is not None:
                os.close(leaf_fd)
            for fd in reversed(parents):
                os.close(fd)

    def _copy_exact(
        self, relative_path: str, private_path: str, *,
        expected_size: int, expected_sha256: str, maximum: int,
        volume_source: bool,
    ) -> None:
        _copy_bounds(expected_size, expected_sha256, maximum)
        private_parts = _parts(private_path, absolute=True)
        try:
            self._check_root()
        except (OSError, VolumeRootBindingError):
            raise _invalid("COPY_ROOT_ADMISSION") from None
        private_chain = None
        member_parents: list[int] = []
        source_fd = None
        destination_fd = None
        destination_identity = None
        source_identity = None
        success = False
        failure = "SOURCE_CHAIN_OPEN"
        try:
            private_chain = _private_chain(private_parts[:-1])
            failure = "COPY_MEMBER_PARENT"
            member_parent, member_leaf, member_parents = self._member_parent(relative_path)
            private_parent, private_leaf = private_chain.descriptors[-1], private_parts[-1]
            if volume_source:
                source_parent, source_leaf = member_parent, member_leaf
                destination_parent, destination_leaf = private_parent, private_leaf
            else:
                source_parent, source_leaf = private_parent, private_leaf
                destination_parent, destination_leaf = member_parent, member_leaf
            failure = "COPY_SOURCE_ADMISSION"
            source_fd, source_identity = _open_exact_source(source_parent, source_leaf, expected_size)
            failure = "COPY_DEST_CREATE_OS"
            try:
                destination_fd = os.open(destination_leaf, _WRITE, 0o600, dir_fd=destination_parent)
            except OSError as error:
                raise _invalid(_create_code("COPY_DEST_CREATE", error)) from None
            failure = "COPY_DEST_IDENTITY"
            destination_info = os.fstat(destination_fd)
            if not stat.S_ISREG(destination_info.st_mode) or destination_info.st_nlink != 1:
                raise _invalid("COPY_DEST_IDENTITY")
            destination_identity = _identity(destination_info)
            failure = "COPY_STREAM"
            _stream_exact(source_fd, destination_fd, expected_size, expected_sha256)
            failure = "COPY_SOURCE_RECHECK"
            if _file_identity(os.fstat(source_fd)) != source_identity or _file_identity(
                os.stat(source_leaf, dir_fd=source_parent, follow_symlinks=False)
            ) != source_identity:
                raise _invalid("COPY_SOURCE_RECHECK")
            failure = "COPY_DEST_RECHECK"
            final_destination = os.fstat(destination_fd)
            named_destination = os.stat(destination_leaf, dir_fd=destination_parent, follow_symlinks=False)
            if (
                not stat.S_ISREG(final_destination.st_mode)
                or final_destination.st_nlink != 1
                or final_destination.st_size != expected_size
                or _identity(final_destination) != destination_identity
                or _identity(named_destination) != destination_identity
            ):
                raise _invalid("COPY_DEST_RECHECK")
            failure = "COPY_MEMBER_RECHECK"
            self._check_member_parents(relative_path, member_parents)
            failure = "COPY_PRIVATE_RECHECK"
            _recheck_chain(private_chain)
            failure = "COPY_ROOT_RECHECK"
            self._check_root()
            success = True
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:
            code = error.code if type(error) is VolumeRootBindingError else None
            raise _invalid(code or failure) from None
        finally:
            for fd in (source_fd, destination_fd):
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            if not success and volume_source and destination_identity is not None:
                try:
                    self._check_member_parents(relative_path, member_parents)
                    _recheck_chain(private_chain)
                    self._check_root()
                    _remove_owned_leaf(destination_parent, destination_leaf, destination_identity)
                except BaseException:
                    pass
            for fd in reversed(member_parents):
                try:
                    os.close(fd)
                except OSError:
                    pass
            if private_chain is not None:
                try:
                    _close_all(private_chain.descriptors)
                except VolumeRootBindingError:
                    pass

    def copy_in_exclusive(
        self, relative_path: str, source_path: str, *,
        expected_size: int, expected_sha256: str, maximum: int,
    ) -> None:
        """Copy a private regular source into the Volume with an exclusive leaf."""
        self._copy_exact(
            relative_path, source_path, expected_size=expected_size,
            expected_sha256=expected_sha256, maximum=maximum, volume_source=False,
        )

    def copy_out(
        self, relative_path: str, destination_path: str, *,
        expected_size: int, expected_sha256: str, maximum: int,
    ) -> None:
        """Copy a bound regular member to an exclusive private destination."""
        self._copy_exact(
            relative_path, destination_path, expected_size=expected_size,
            expected_sha256=expected_sha256, maximum=maximum, volume_source=True,
        )

    def write_exclusive(self, path: str, content: bytes, maximum: int) -> None:
        if type(content) is not bytes or type(maximum) is not int or not 0 <= len(content) <= maximum <= 1_073_741_824:
            raise _invalid()
        self._check_root()
        parents: list[int] = []
        leaf_fd = None
        try:
            parent, leaf, parents = self._member_parent(path)
            leaf_fd = os.open(leaf, _WRITE, 0o600, dir_fd=parent)
            info = os.fstat(leaf_fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise _invalid()
            view = memoryview(content)
            while view:
                written = os.write(leaf_fd, view)
                if written <= 0:
                    raise _invalid()
                view = view[written:]
            os.fsync(leaf_fd)
            if os.fstat(leaf_fd).st_size != len(content):
                raise _invalid()
            self._check_member_parents(path, parents)
            self._check_root()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise _invalid() from None
        finally:
            if leaf_fd is not None:
                os.close(leaf_fd)
            for fd in reversed(parents):
                os.close(fd)

    def claim_directory(self, path: str) -> None:
        """Create one directory exclusively under the retained root."""
        try:
            self._check_root()
        except (OSError, VolumeRootBindingError):
            raise _invalid("CLAIM_ROOT_ADMISSION") from None
        parents: list[int] = []
        failure = "CLAIM_PARENT"
        try:
            parent, leaf, parents = self._member_parent(path)
            failure = "CLAIM_CREATE_OS"
            try:
                os.mkdir(leaf, 0o700, dir_fd=parent)
            except OSError as error:
                raise _invalid(_create_code("CLAIM_CREATE", error)) from None
            failure = "CLAIM_IDENTITY"
            created = os.open(leaf, _DIR, dir_fd=parent)
            try:
                info = os.fstat(created)
                named = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
                if not stat.S_ISDIR(info.st_mode) or _identity(info) != _identity(named):
                    raise _invalid("CLAIM_IDENTITY")
                failure = "CLAIM_RECHECK"
                self._check_member_parents(path, parents)
                self._check_root()
                named = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
                if _identity(info) != _identity(named):
                    raise _invalid("CLAIM_RECHECK")
            finally:
                os.close(created)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:
            code = error.code if type(error) is VolumeRootBindingError else None
            raise _invalid(code or failure) from None
        finally:
            for fd in reversed(parents):
                os.close(fd)

    def list_regular(self, path: str, maximum_entries: int) -> tuple[tuple[str, int], ...]:
        if type(maximum_entries) is not int or not 0 <= maximum_entries <= 100_000:
            raise _invalid()
        self._check_root()
        parents: list[int] = []
        directory = None
        try:
            if path == "":
                directory = os.dup(self._target.descriptors[-1])
            else:
                parent, leaf, parents = self._member_parent(path)
                directory = os.open(leaf, _DIR, dir_fd=parent)
            result = []
            with os.scandir(directory) as entries:
                for entry in entries:
                    if len(result) >= maximum_entries:
                        raise _invalid()
                    info = entry.stat(follow_symlinks=False)
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                        raise _invalid()
                    result.append((entry.name, info.st_size))
            if path:
                self._check_member_parents(path, parents)
                reopened = os.open(leaf, _DIR, dir_fd=parent)
                try:
                    if _identity(os.fstat(reopened)) != _identity(os.fstat(directory)):
                        raise _invalid()
                finally:
                    os.close(reopened)
            self._check_root()
            return tuple(sorted(result))
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise _invalid() from None
        finally:
            if directory is not None:
                os.close(directory)
            for fd in reversed(parents):
                os.close(fd)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            _close_all([*self._root_parent.descriptors, *self._target.descriptors, self._link_fd])

    def __enter__(self) -> "VolumeRootBinding":
        try:
            self._check_root()
        except BaseException:
            try:
                self.close()
            except VolumeRootBindingError:
                pass
            raise
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["VolumeRootBinding", "VolumeRootBindingError"]
