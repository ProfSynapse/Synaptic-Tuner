"""Private content-addressed publication for prepared datasets.

The publication protocol authenticates and retains filesystem authorities and
serializes cooperating publishers that use this API.  Active subversion by a
process running as the same OS identity, kernel compromise, and manipulation of
already-retained handles are outside its threat model.  As with the tokenizer
boundary, callers must not grant untrusted code the publisher's OS authority.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import secrets
import stat
import sys
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn

from tuner.ingestion import load_verified_normalized_bundle_v1

from .models import (
    ARTIFACT_SCHEMA_VERSION,
    CONFIG_SCHEMA_VERSION,
    RAW_TEXT_FORMAT,
    ROW_SCHEMA_VERSION,
    DatasetPrepCollisionError,
    DatasetPrepConfigV1,
    DatasetPrepDurabilityError,
    DatasetPrepValidationError,
    DatasetPublicationUncertainV1,
    DatasetPublicationUncertaintyPhaseV1,
    DatasetSemanticIdentityV1,
    ProjectionV1,
    VerifiedPreparedDatasetV1,
)
from .prepare import (
    MAX_DATASET_BYTES,
    _DATASET_DOMAIN,
    _PROJECTION_DOMAIN,
    _ROW_ID_DOMAIN,
    _canonical_bytes,
    _domain_digest,
    _stream_digest,
    build_prepared_dataset_v1,
)
from .context_messages import (
    ARTIFACT_SCHEMA_VERSION_V2,
    CONFIG_SCHEMA_VERSION_V2,
    MESSAGES_FORMAT,
    ROW_SCHEMA_VERSION_V2,
    DatasetPrepConfigV2,
    build_prepared_dataset_v2,
    _DATASET_DOMAIN_V2,
    _LINEAGE_DOMAIN_V2,
    _PROJECTION_DOMAIN_V2,
    _ROW_ID_DOMAIN_V2,
)


MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_ROWS = 100_000
_EXPECTED_INVENTORY = frozenset({"manifest.json", "dataset.jsonl"})
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ROW_ID = re.compile(r"^row-[0-9a-f]{64}$")
_ITEM_ID = re.compile(r"^item-[0-9a-f]{64}$")
_PATH_TYPE = type(Path())
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version", "dataset_id", "dataset_digest", "source", "format",
        "row_schema_version", "projection", "projection_digest", "recipe",
        "row_count", "dataset_bytes", "dataset_sha256", "row_ids_sha256",
        "split_counts",
    }
)
_ROW_FIELDS = frozenset({"schema_version", "format", "row_id", "source_item_id", "split", "text"})
_MANIFEST_FIELDS_V2 = frozenset(
    {
        "schema_version", "dataset_id", "dataset_digest", "source", "format",
        "row_schema_version", "projection", "projection_digest", "recipe",
        "lineage", "lineage_digest", "group_count", "group_ids_sha256",
        "row_count", "dataset_bytes", "dataset_sha256", "row_ids_sha256",
        "split_counts",
    }
)
_ROW_FIELDS_V2 = frozenset(
    {
        "schema_version", "format", "row_id", "target_item_id",
        "context_item_ids", "group_id", "split", "messages",
    }
)
_LINEAGE_FIELDS_V2 = frozenset(
    {
        "row_id", "target_item_id", "context_item_ids", "group_id",
        "prompt_variant", "prompt_template_sha256", "separator_sha256",
        "rendered_user_sha256", "assistant_sha256",
    }
)


class _PostRenameIdentityMismatch(RuntimeError):
    __slots__ = ("scrubbed",)

    def __init__(self, scrubbed: bool) -> None:
        self.scrubbed = scrubbed
        super().__init__()


def _invalid(message: str = "prepared dataset is invalid") -> NoReturn:
    raise DatasetPrepValidationError(message) from None


def _exact(value: object, fields: frozenset[str], name: str) -> dict[str, object]:
    if type(value) is not dict or frozenset(value) != fields:
        _invalid(f"{name} has missing or unknown fields")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _invalid("canonical JSON contains duplicate keys")
        result[key] = value
    return result


def _parse_canonical_object(raw: bytes, maximum: int, name: str) -> dict[str, object]:
    if type(raw) is not bytes or not raw or len(raw) > maximum:
        _invalid(f"{name} is not bounded canonical JSON")
    depth = 0
    nodes = 1
    in_string = False
    escaped = False
    for byte in raw:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte in (0x7B, 0x5B):
            depth += 1
            nodes += 1
            if depth > 32:
                _invalid(f"{name} exceeds the nesting limit")
        elif byte in (0x7D, 0x5D):
            depth -= 1
            if depth < 0:
                _invalid(f"{name} has invalid JSON structure")
        elif byte in (0x2C, 0x3A):
            nodes += 1
            if nodes > 500_000:
                _invalid(f"{name} exceeds the node limit")
    if in_string or depth != 0:
        _invalid(f"{name} has invalid JSON structure")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _value: _invalid(f"{name} contains a non-finite number"),
        )
    except DatasetPrepValidationError:
        raise
    except BaseException:
        _invalid(f"{name} is not canonical UTF-8 JSON")
    if type(value) is not dict or _canonical_bytes(value) != raw:
        _invalid(f"{name} is not a canonical JSON object")
    return value


def _is_reparse(info: os.stat_result) -> bool:
    return bool(getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, getattr(info, "st_mtime_ns", 0))


def _same_file(
    left: tuple[int, int, int, int, int],
    right: tuple[int, int, int, int, int],
) -> bool:
    return (left[0], left[1], left[3]) == (right[0], right[1], right[3])


def _same_directory(
    left: tuple[int, int, int, int, int],
    right: tuple[int, int, int, int, int],
) -> bool:
    return (left[0], left[1], stat.S_IFMT(left[2])) == (
        right[0], right[1], stat.S_IFMT(right[2])
    )


if os.name == "nt":
    from ctypes import wintypes

    class _WinFileInfo(ctypes.Structure):
        _fields_ = [
            ("attributes", wintypes.DWORD),
            ("creation_time", wintypes.FILETIME),
            ("access_time", wintypes.FILETIME),
            ("write_time", wintypes.FILETIME),
            ("volume_serial", wintypes.DWORD),
            ("size_high", wintypes.DWORD),
            ("size_low", wintypes.DWORD),
            ("links", wintypes.DWORD),
            ("file_index_high", wintypes.DWORD),
            ("file_index_low", wintypes.DWORD),
        ]


def _win_open_directory(path: Path, *, deletable: bool) -> int:
    if os.name != "nt":
        raise RuntimeError("Windows directory handles are unavailable")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = ctypes.c_void_p
    access = 0x80 | 0x100000 | (0x10000 if deletable else 0)  # READ_ATTRIBUTES|SYNCHRONIZE|DELETE
    handle = kernel32.CreateFileW(
        str(path), access, 0x1 | 0x2, None, 3, 0x02000000 | 0x00200000, None
    )
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid or handle in (0, None):
        raise OSError(ctypes.get_last_error(), "directory handle could not be retained")
    return int(handle)


def _win_open_file_for_delete(path: Path) -> int:
    if os.name != "nt":
        raise RuntimeError("Windows file handles are unavailable")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = ctypes.c_void_p
    handle = kernel32.CreateFileW(
        str(path), 0x80 | 0x100000 | 0x10000, 0x1 | 0x2, None, 3, 0x00200000, None
    )
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid or handle in (0, None):
        raise OSError(ctypes.get_last_error(), "file handle could not be retained")
    return int(handle)


def _win_handle_identity(handle: int) -> tuple[int, int]:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
    info = _WinFileInfo()
    if not kernel32.GetFileInformationByHandle(ctypes.c_void_p(handle), ctypes.byref(info)):
        raise OSError(ctypes.get_last_error(), "directory handle identity is unavailable")
    return info.volume_serial, (info.file_index_high << 32) | info.file_index_low


def _win_close_handle(handle: int) -> bool:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.restype = wintypes.BOOL
    return bool(kernel32.CloseHandle(ctypes.c_void_p(handle)))


def _win_acquire_root_mutex(identity: tuple[int, int, int, int, int]) -> int:
    if os.name != "nt":
        raise RuntimeError("Windows coordination mutexes are unavailable")
    token = hashlib.sha256(f"{identity[0]}:{identity[1]}".encode("ascii")).hexdigest()
    name = f"Local\\SyntuniaDatasetPrep-{token}"
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    handle = kernel32.CreateMutexW(None, False, name)
    if handle in (0, None):
        raise OSError(ctypes.get_last_error(), "publication coordination is unavailable")
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    result = kernel32.WaitForSingleObject(ctypes.c_void_p(handle), 0xFFFFFFFF)
    if result not in (0, 0x80):  # WAIT_OBJECT_0 or WAIT_ABANDONED
        _win_close_handle(int(handle))
        raise OSError(ctypes.get_last_error(), "publication coordination could not be acquired")
    return int(handle)


def _win_release_root_mutex(handle: int) -> bool:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.ReleaseMutex.restype = wintypes.BOOL
    released = bool(kernel32.ReleaseMutex(ctypes.c_void_p(handle)))
    return _win_close_handle(handle) and released


def _win_rename_directory_relative(handle: int, root_handle: int, destination_name: str) -> None:
    encoded = destination_name.encode("utf-16-le")
    pointer_size = ctypes.sizeof(ctypes.c_void_p)
    root_offset = 8 if pointer_size == 8 else 4
    length_offset = root_offset + pointer_size
    name_offset = length_offset + 4
    buffer = ctypes.create_string_buffer(name_offset + len(encoded) + 2)
    ctypes.c_int.from_buffer(buffer, 0).value = 0
    ctypes.c_void_p.from_buffer(buffer, root_offset).value = root_handle
    ctypes.c_uint32.from_buffer(buffer, length_offset).value = len(encoded)
    ctypes.memmove(ctypes.addressof(buffer) + name_offset, encoded, len(encoded))
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.SetFileInformationByHandle.restype = wintypes.BOOL
    if not kernel32.SetFileInformationByHandle(
        ctypes.c_void_p(handle), 22, ctypes.byref(buffer), len(buffer)
    ):
        code = ctypes.get_last_error()
        if code in (80, 183):
            raise FileExistsError(errno.EEXIST, "destination exists")
        raise OSError(code, "directory rename failed")


def _win_mark_directory_delete(handle: int) -> None:
    delete = ctypes.c_int(1)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.SetFileInformationByHandle.restype = wintypes.BOOL
    if not kernel32.SetFileInformationByHandle(
        ctypes.c_void_p(handle), 4, ctypes.byref(delete), ctypes.sizeof(delete)
    ):
        raise OSError(ctypes.get_last_error(), "directory delete claim failed")


def _win_delete_member_exact(path: Path, expected: tuple[int, int, int, int, int]) -> None:
    handle = _win_open_file_for_delete(path)
    failed = False
    try:
        if _win_handle_identity(handle)[1] != expected[1]:
            _invalid("owned stage member identity changed")
        current = _identity(path.lstat())
        if not _same_file(current, expected):
            _invalid("owned stage member identity changed")
        _win_mark_directory_delete(handle)
    except BaseException:
        failed = True
    finally:
        if not _win_close_handle(handle):
            failed = True
    if failed:
        _invalid("owned stage member could not be removed safely")


def _plain_directory(path: Path) -> tuple[int, int, int, int, int]:
    try:
        info = path.lstat()
    except OSError:
        _invalid("prepared dataset directory is unavailable")
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode) or _is_reparse(info):
        _invalid("prepared dataset directory is not a plain directory")
    try:
        if path.resolve(strict=True) != path.absolute():
            _invalid("prepared dataset directory traverses a link")
    except DatasetPrepValidationError:
        raise
    except BaseException:
        _invalid("prepared dataset directory is unavailable")
    return _identity(info)


def _inventory(path: Path, directory_identity: tuple[int, int, int, int, int]) -> dict[str, tuple[int, int, int, int, int]]:
    if _plain_directory(path) != directory_identity:
        _invalid("prepared dataset directory identity changed")
    result: dict[str, tuple[int, int, int, int, int]] = {}
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if len(result) == len(_EXPECTED_INVENTORY):
                    _invalid("prepared dataset inventory is invalid")
                if (
                    type(entry.name) is not str
                    or entry.name not in _EXPECTED_INVENTORY
                    or entry.name in result
                ):
                    _invalid("prepared dataset inventory is invalid")
                # Windows DirEntry.stat() may report zero device/inode values
                # while opening the same member yields its real identity.
                info = (path / entry.name).lstat()
                if (
                    not stat.S_ISREG(info.st_mode)
                    or stat.S_ISLNK(info.st_mode)
                    or _is_reparse(info)
                    or info.st_nlink != 1
                ):
                    _invalid("prepared dataset member is not a plain file")
                result[entry.name] = _identity(info)
    except DatasetPrepValidationError:
        raise
    except BaseException:
        _invalid("prepared dataset inventory is unavailable")
    if frozenset(result) != _EXPECTED_INVENTORY:
        _invalid("prepared dataset inventory is invalid")
    if _plain_directory(path) != directory_identity:
        _invalid("prepared dataset directory identity changed")
    return result


class _RetainedDatasetDirectory:
    __slots__ = ("path", "identity", "member_identities")

    def __init__(self, path: Path) -> None:
        self.path = path
        self.identity = _plain_directory(path)
        self.member_identities = _inventory(path, self.identity)

    def verify_final_inventory(self) -> None:
        if _inventory(self.path, self.identity) != self.member_identities:
            _invalid("prepared dataset inventory changed during verification")


class _RetainedMember:
    __slots__ = ("path", "maximum", "handle", "opened_identity", "closed")

    def __init__(
        self,
        path: Path,
        maximum: int,
        expected_identity: tuple[int, int, int, int, int],
    ) -> None:
        self.path = path
        self.maximum = maximum
        self.handle = None
        self.opened_identity = expected_identity
        self.closed = False
        try:
            declared = path.lstat()
            declared_identity = _identity(declared)
            if declared_identity != expected_identity:
                _invalid("prepared dataset member identity changed")
            if (
                stat.S_ISLNK(declared.st_mode)
                or _is_reparse(declared)
                or not stat.S_ISREG(declared.st_mode)
                or declared.st_nlink != 1
            ):
                _invalid("prepared dataset member is not a plain file")
            if declared.st_size > maximum or path.resolve(strict=True) != path.absolute():
                _invalid("prepared dataset member is not a bounded plain file")
            flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            try:
                self.handle = os.fdopen(descriptor, "rb", buffering=0)
            except BaseException:
                os.close(descriptor)
                raise
            opened_info = os.fstat(self.handle.fileno())
            opened = _identity(opened_info)
            if opened_info.st_nlink != 1:
                _invalid("prepared dataset member is hardlinked")
            if not _same_file(declared_identity, opened):
                _invalid("prepared dataset member identity changed")
            if opened[3] > maximum:
                _invalid("prepared dataset member exceeds its byte limit")
            if os.name != "nt" and stat.S_IMODE(opened[2]) & 0o077:
                _invalid("prepared dataset member permissions are not private")
            self.opened_identity = opened
        except DatasetPrepValidationError:
            self.close()
            raise
        except BaseException:
            self.close()
            _invalid("prepared dataset member could not be retained safely")

    def read(self) -> bytes:
        if self.handle is None or self.closed:
            _invalid("prepared dataset member handle is unavailable")
        try:
            chunks: list[bytes] = []
            size = 0
            while size <= self.maximum:
                chunk = self.handle.read(min(1024 * 1024, self.maximum + 1 - size))
                if not chunk:
                    break
                if type(chunk) is not bytes:
                    _invalid("prepared dataset member read returned invalid bytes")
                chunks.append(chunk)
                size += len(chunk)
            payload = b"".join(chunks)
        except DatasetPrepValidationError:
            raise
        except BaseException:
            _invalid("prepared dataset member could not be read safely")
        if len(payload) > self.maximum or len(payload) != self.opened_identity[3]:
            _invalid("prepared dataset member size changed during verification")
        return payload

    def verify_final_identity(self) -> None:
        if self.handle is None or self.closed:
            _invalid("prepared dataset member handle is unavailable")
        try:
            final_info = os.fstat(self.handle.fileno())
            final = _identity(final_info)
        except BaseException:
            _invalid("prepared dataset member identity could not be revalidated")
        if final != self.opened_identity or final_info.st_nlink != 1:
            _invalid("prepared dataset member changed during verification")

    def close(self) -> bool:
        if self.closed or self.handle is None:
            return True
        failed = False
        try:
            self.handle.close()
        except BaseException:
            failed = True
        self.closed = True
        return not failed


class _RetainedRelativeMember:
    """A member retained relative to an authenticated POSIX directory descriptor."""

    __slots__ = ("name", "maximum", "handle", "opened_identity", "closed")

    def __init__(
        self,
        directory_fd: int,
        name: str,
        maximum: int,
        expected_identity: tuple[int, int, int, int, int],
    ) -> None:
        self.name = name
        self.maximum = maximum
        self.handle = None
        self.opened_identity = expected_identity
        self.closed = False
        try:
            declared = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if _identity(declared) != expected_identity:
                _invalid("prepared dataset member identity changed")
            if (
                stat.S_ISLNK(declared.st_mode)
                or not stat.S_ISREG(declared.st_mode)
                or declared.st_nlink != 1
            ):
                _invalid("prepared dataset member is not a plain file")
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                self.handle = os.fdopen(descriptor, "rb", buffering=0)
            except BaseException:
                os.close(descriptor)
                raise
            opened_info = os.fstat(self.handle.fileno())
            opened = _identity(opened_info)
            if not _same_file(expected_identity, opened) or opened[3] > maximum:
                _invalid("prepared dataset member identity changed")
            if opened_info.st_nlink != 1:
                _invalid("prepared dataset member is hardlinked")
            if stat.S_IMODE(opened[2]) & 0o077:
                _invalid("prepared dataset member permissions are not private")
            self.opened_identity = opened
        except DatasetPrepValidationError:
            self.close()
            raise
        except BaseException:
            self.close()
            _invalid("prepared dataset member could not be retained safely")

    read = _RetainedMember.read
    verify_final_identity = _RetainedMember.verify_final_identity
    close = _RetainedMember.close


class _OutputRootAuthority:
    __slots__ = ("path", "identity", "descriptor", "native_handle", "closed")

    def __init__(self, path: Path) -> None:
        self.path = path
        self.identity = _plain_directory(path)
        self.descriptor: int | None = None
        self.native_handle: int | None = None
        self.closed = False
        if os.name == "nt":
            try:
                self.native_handle = _win_open_directory(path, deletable=False)
                if _win_handle_identity(self.native_handle)[1] != self.identity[1]:
                    _win_close_handle(self.native_handle)
                    self.native_handle = None
                    _invalid("prepared dataset output root identity changed")
            except DatasetPrepValidationError:
                raise
            except BaseException:
                _invalid("prepared dataset output root could not be retained safely")
        else:
            try:
                descriptor = os.open(
                    path,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                )
                if not _same_directory(_identity(os.fstat(descriptor)), self.identity):
                    os.close(descriptor)
                    _invalid("prepared dataset output root identity changed")
                self.descriptor = descriptor
            except DatasetPrepValidationError:
                raise
            except BaseException:
                _invalid("prepared dataset output root could not be retained safely")

    def verify(self) -> None:
        if self.closed:
            _invalid("prepared dataset output root authority is unavailable")
        if self.descriptor is not None:
            try:
                if not _same_directory(_identity(os.fstat(self.descriptor)), self.identity):
                    _invalid("prepared dataset output root identity changed")
            except DatasetPrepValidationError:
                raise
            except BaseException:
                _invalid("prepared dataset output root could not be revalidated")
        if self.native_handle is not None:
            try:
                if _win_handle_identity(self.native_handle)[1] != self.identity[1]:
                    _invalid("prepared dataset output root identity changed")
            except DatasetPrepValidationError:
                raise
            except BaseException:
                _invalid("prepared dataset output root could not be revalidated")
        if not _same_directory(_plain_directory(self.path), self.identity):
            _invalid("prepared dataset output root identity changed")

    def close(self) -> bool:
        if self.closed:
            return True
        failed = False
        if self.descriptor is not None:
            try:
                os.close(self.descriptor)
            except OSError:
                failed = True
        if self.native_handle is not None and not _win_close_handle(self.native_handle):
            failed = True
        self.closed = True
        return not failed


class _OutputRootCoordination:
    """Exclusive cooperative lock bound to an authenticated retained root."""

    __slots__ = ("root", "native_handle", "closed")

    def __init__(self, root: _OutputRootAuthority) -> None:
        self.root = root
        self.native_handle: int | None = None
        self.closed = False
        root.verify()
        try:
            if root.descriptor is not None:
                import fcntl

                fcntl.flock(root.descriptor, fcntl.LOCK_EX)
            else:
                self.native_handle = _win_acquire_root_mutex(root.identity)
            root.verify()
        except DatasetPrepValidationError:
            self.close()
            raise
        except BaseException:
            self.close()
            _invalid("prepared dataset publication coordination is unavailable")

    def close(self) -> bool:
        if self.closed:
            return True
        failed = False
        if self.root.descriptor is not None:
            try:
                import fcntl

                fcntl.flock(self.root.descriptor, fcntl.LOCK_UN)
            except BaseException:
                failed = True
        if self.native_handle is not None and not _win_release_root_mutex(self.native_handle):
            failed = True
        self.native_handle = None
        self.closed = True
        return not failed


class _OwnedStageAuthority:
    __slots__ = (
        "root", "name", "path", "identity", "descriptor", "native_handle",
        "closed", "published_name", "committed", "cleaned",
    )

    def __init__(self, root: _OutputRootAuthority, name: str, path: Path, identity, descriptor: int | None) -> None:
        self.root = root
        self.name = name
        self.path = path
        self.identity = identity
        self.descriptor = descriptor
        self.native_handle: int | None = None
        self.closed = False
        self.published_name: str | None = None
        self.committed = False
        self.cleaned = False

    @classmethod
    def create(cls, root: _OutputRootAuthority, dataset_id: str) -> "_OwnedStageAuthority":
        root.verify()
        for _attempt in range(32):
            name = f".{dataset_id}.{secrets.token_hex(16)}"
            path = root.path / name
            try:
                if root.descriptor is not None:
                    os.mkdir(name, 0o700, dir_fd=root.descriptor)
                    descriptor = os.open(
                        name,
                        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=root.descriptor,
                    )
                    identity = _identity(os.fstat(descriptor))
                    declared = os.stat(name, dir_fd=root.descriptor, follow_symlinks=False)
                    if not _same_directory(_identity(declared), identity) or not stat.S_ISDIR(declared.st_mode):
                        os.close(descriptor)
                        _invalid("prepared dataset stage identity changed")
                    return cls(root, name, path, identity, descriptor)
                path.mkdir(mode=0o700)
                os.chmod(path, 0o700)
                authority = cls(root, name, path, _plain_directory(path), None)
                authority.native_handle = _win_open_directory(path, deletable=True)
                if _win_handle_identity(authority.native_handle)[1] != authority.identity[1]:
                    authority.close()
                    _invalid("prepared dataset stage identity changed")
                return authority
            except FileExistsError:
                continue
            except DatasetPrepValidationError:
                raise
            except BaseException:
                _invalid("prepared dataset stage could not be created safely")
        _invalid("prepared dataset stage name could not be allocated")

    def verify_binding(self) -> None:
        self.root.verify()
        if self.closed:
            _invalid("prepared dataset stage authority is unavailable")
        if self.descriptor is not None:
            try:
                retained = _identity(os.fstat(self.descriptor))
                declared = _identity(os.stat(self.name, dir_fd=self.root.descriptor, follow_symlinks=False))
            except BaseException:
                _invalid("prepared dataset stage binding is unavailable")
            if not _same_directory(retained, self.identity) or not _same_directory(declared, self.identity):
                _invalid("prepared dataset stage binding changed")
        elif not _same_directory(_plain_directory(self.path), self.identity):
            _invalid("prepared dataset stage binding changed")
        if self.native_handle is not None:
            try:
                if _win_handle_identity(self.native_handle)[1] != self.identity[1]:
                    _invalid("prepared dataset stage binding changed")
            except DatasetPrepValidationError:
                raise
            except BaseException:
                _invalid("prepared dataset stage binding is unavailable")

    def close(self) -> bool:
        if self.closed:
            return True
        failed = False
        if self.descriptor is not None:
            try:
                os.close(self.descriptor)
            except OSError:
                failed = True
        if self.native_handle is not None and not _win_close_handle(self.native_handle):
            failed = True
        self.native_handle = None
        self.closed = True
        return not failed


def _positive_int(value: object, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        _invalid(f"{name} is invalid")
    return value


def _nonnegative_int(value: object, name: str, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        _invalid(f"{name} is invalid")
    return value


def _digest(value: object, name: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        _invalid(f"{name} is invalid")
    return value


def _validate_recipe(value: object) -> dict[str, object]:
    recipe = _exact(value, frozenset({"schema_version", "format", "projection", "ordering", "split"}), "recipe")
    if recipe["schema_version"] != CONFIG_SCHEMA_VERSION or recipe["format"] != RAW_TEXT_FORMAT:
        _invalid("recipe version or format is unsupported")
    ProjectionV1.from_dict(recipe["projection"])
    ordering = recipe["ordering"]
    if type(ordering) is not dict or ordering.get("kind") not in {"source_order", "seeded_hash"}:
        _invalid("recipe ordering is invalid")
    ordering_fields = frozenset({"kind"}) if ordering["kind"] == "source_order" else frozenset({"kind", "seed_sha256"})
    ordering = _exact(ordering, ordering_fields, "recipe ordering")
    if "seed_sha256" in ordering:
        _digest(ordering["seed_sha256"], "ordering seed digest")
    split = recipe["split"]
    if type(split) is not dict or split.get("kind") not in {"none", "hash_rank"}:
        _invalid("recipe split is invalid")
    split_fields = frozenset({"kind"}) if split["kind"] == "none" else frozenset({"kind", "seed_sha256", "allocations"})
    split = _exact(split, split_fields, "recipe split")
    if split["kind"] == "hash_rank":
        _digest(split["seed_sha256"], "split seed digest")
        allocations = split["allocations"]
        if type(allocations) is not list or not 2 <= len(allocations) <= 32:
            _invalid("recipe split allocations are invalid")
        names: list[str] = []
        for raw in allocations:
            item = _exact(raw, frozenset({"name", "weight"}), "recipe split allocation")
            name = item["name"]
            if type(name) is not str or not name:
                _invalid("recipe split allocation name is invalid")
            _positive_int(item["weight"], "recipe split allocation weight", 2**31 - 1)
            names.append(name)
        if names != ["train", "validation"]:
            _invalid("recipe split allocation names are invalid")
    return recipe


def _verify_bytes_v1(
    path: Path,
    manifest_raw: bytes,
    dataset_raw: bytes,
    *,
    require_content_addressed_name: bool = True,
) -> VerifiedPreparedDatasetV1:
    manifest = _exact(_parse_canonical_object(manifest_raw, MAX_MANIFEST_BYTES, "manifest"), _MANIFEST_FIELDS, "manifest")
    if manifest["schema_version"] != ARTIFACT_SCHEMA_VERSION or manifest["format"] != RAW_TEXT_FORMAT:
        _invalid("manifest version or format is unsupported")
    if manifest["row_schema_version"] != ROW_SCHEMA_VERSION:
        _invalid("row schema version is unsupported")
    dataset_digest = _digest(manifest["dataset_digest"], "dataset_digest")
    dataset_id = manifest["dataset_id"]
    if (
        type(dataset_id) is not str
        or dataset_id != "dataset-" + dataset_digest
        or (require_content_addressed_name and path.name != dataset_id)
    ):
        _invalid("dataset identity is invalid")
    source = _exact(manifest["source"], frozenset({"bundle_digest", "structure_set_digest", "item_count"}), "source")
    bundle_digest = _digest(source["bundle_digest"], "bundle_digest")
    structure_set_digest = _digest(source["structure_set_digest"], "structure_set_digest")
    item_count = _positive_int(source["item_count"], "item_count", MAX_ROWS)
    projection = _exact(manifest["projection"], frozenset({"structure_ref", "name", "field_ref"}), "projection")
    projection_config = ProjectionV1.from_dict({"structure_ref": projection["structure_ref"], "name": projection["name"]})
    field_ref = projection["field_ref"]
    if type(field_ref) is not str or not field_ref:
        _invalid("projection field_ref is invalid")
    projection_digest = _digest(manifest["projection_digest"], "projection_digest")
    if _domain_digest(_PROJECTION_DOMAIN, projection) != projection_digest:
        _invalid("projection digest is invalid")
    recipe = _validate_recipe(manifest["recipe"])
    if recipe["projection"] != projection_config.to_dict():
        _invalid("recipe projection does not match resolved projection")
    row_count = _positive_int(manifest["row_count"], "row_count", MAX_ROWS)
    dataset_bytes = _positive_int(manifest["dataset_bytes"], "dataset_bytes", MAX_DATASET_BYTES)
    dataset_sha256 = _digest(manifest["dataset_sha256"], "dataset_sha256")
    row_ids_sha256 = _digest(manifest["row_ids_sha256"], "row_ids_sha256")
    split_counts = manifest["split_counts"]
    if type(split_counts) is not dict or not split_counts:
        _invalid("split_counts is invalid")
    checked_counts: dict[str, int] = {}
    for key, value in split_counts.items():
        if type(key) is not str or not key:
            _invalid("split_counts is invalid")
        checked_counts[key] = _nonnegative_int(value, "split count", MAX_ROWS)
    if sum(checked_counts.values()) != row_count or item_count != row_count:
        _invalid("dataset counts are inconsistent")
    split_recipe = recipe["split"]
    if split_recipe["kind"] == "none":
        if checked_counts != {"train": row_count}:
            _invalid("none split counts are invalid")
    else:
        allocations = split_recipe["allocations"]
        total_weight = sum(item["weight"] for item in allocations)
        expected_counts = [row_count * item["weight"] // total_weight for item in allocations]
        remainders = [row_count * item["weight"] % total_weight for item in allocations]
        remaining = row_count - sum(expected_counts)
        for index in sorted(range(len(allocations)), key=lambda item: (-remainders[item], item))[:remaining]:
            expected_counts[index] += 1
        if checked_counts != {
            item["name"]: expected_counts[index] for index, item in enumerate(allocations)
        }:
            _invalid("hash_rank split counts are invalid")
    if len(dataset_raw) != dataset_bytes or hashlib.sha256(dataset_raw).hexdigest() != dataset_sha256:
        _invalid("dataset bytes do not match manifest")
    if not dataset_raw.endswith(b"\n"):
        _invalid("dataset JSONL is not newline terminated")
    lines = dataset_raw[:-1].split(b"\n")
    if len(lines) != row_count or any(not line for line in lines):
        _invalid("dataset row count is invalid")
    row_ids: list[str] = []
    observed_counts: dict[str, int] = {name: 0 for name in checked_counts}
    for line in lines:
        row = _exact(_parse_canonical_object(line, MAX_DATASET_BYTES, "dataset row"), _ROW_FIELDS, "dataset row")
        if row["schema_version"] != ROW_SCHEMA_VERSION or row["format"] != RAW_TEXT_FORMAT:
            _invalid("dataset row version or format is unsupported")
        row_id = row["row_id"]
        source_item_id = row["source_item_id"]
        split = row["split"]
        text = row["text"]
        if type(row_id) is not str or _ROW_ID.fullmatch(row_id) is None:
            _invalid("row_id is invalid")
        if type(source_item_id) is not str or _ITEM_ID.fullmatch(source_item_id) is None:
            _invalid("source_item_id is invalid")
        if type(split) is not str or split not in checked_counts:
            _invalid("row split is invalid")
        if type(text) is not str or not text:
            _invalid("row text is invalid")
        expected_row_id = "row-" + _domain_digest(
            _ROW_ID_DOMAIN,
            {
                "source_bundle_digest": bundle_digest,
                "source_item_id": source_item_id,
                "structure_ref": projection_config.structure_ref.to_dict(),
                "projection": {"name": projection_config.name, "field_ref": field_ref},
                "projected_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "format": RAW_TEXT_FORMAT,
            },
        )
        if row_id != expected_row_id:
            _invalid("row_id does not bind row semantics")
        row_ids.append(row_id)
        observed_counts[split] = observed_counts.get(split, 0) + 1
    if len(row_ids) != len(set(row_ids)) or observed_counts != checked_counts:
        _invalid("dataset row identities or split counts are invalid")
    if _stream_digest(tuple(row_ids)) != row_ids_sha256:
        _invalid("row_ids digest is invalid")
    basis = {
        "source_bundle_digest": bundle_digest,
        "source_structure_set_digest": structure_set_digest,
        "projection": projection,
        "projection_digest": projection_digest,
        "recipe": recipe,
        "row_count": row_count,
        "dataset_bytes": dataset_bytes,
        "dataset_sha256": dataset_sha256,
        "row_ids_sha256": row_ids_sha256,
        "split_counts": checked_counts,
    }
    if _domain_digest(_DATASET_DOMAIN, basis) != dataset_digest:
        _invalid("dataset digest is invalid")
    identity = DatasetSemanticIdentityV1(
        dataset_id, dataset_digest, row_count, dataset_bytes, dataset_sha256,
        row_ids_sha256, MappingProxyType(checked_counts),
    )
    return VerifiedPreparedDatasetV1(path, identity)


def _validate_recipe_v2(value: object) -> dict[str, object]:
    recipe = _exact(
        value,
        frozenset({"schema_version", "format", "target_projection", "context_package", "split"}),
        "v2 recipe",
    )
    if recipe["schema_version"] != CONFIG_SCHEMA_VERSION_V2 or recipe["format"] != MESSAGES_FORMAT:
        _invalid("v2 recipe version or format is unsupported")
    ProjectionV1.from_dict(recipe["target_projection"])
    package = _exact(
        recipe["context_package"],
        frozenset({"ordering", "lineage_digest", "lineage_count", "package_count", "prompt_variants", "target_transforms"}),
        "v2 context_package recipe",
    )
    ordering = _exact(package["ordering"], frozenset({"kind"}), "v2 ordering")
    if ordering["kind"] != "configured_whole_documents":
        _invalid("v2 ordering is unsupported")
    _digest(package["lineage_digest"], "lineage digest")
    _positive_int(package["lineage_count"], "lineage count", MAX_ROWS * 32)
    _positive_int(package["package_count"], "package count", MAX_ROWS)
    variants = package["prompt_variants"]
    if type(variants) is not list or not variants:
        _invalid("v2 prompt variants are invalid")
    names: list[str] = []
    for raw in variants:
        variant = _exact(
            raw,
            frozenset({"name", "prompt_sha256", "separator_sha256"}),
            "v2 prompt variant",
        )
        if type(variant["name"]) is not str or not variant["name"]:
            _invalid("v2 prompt variant name is invalid")
        names.append(variant["name"])
        _digest(variant["prompt_sha256"], "prompt digest")
        _digest(variant["separator_sha256"], "separator digest")
    if len(names) != len(set(names)):
        _invalid("v2 prompt variant names are duplicated")
    transforms = _exact(
        package["target_transforms"],
        frozenset({"kind", "selection_digest", "fenced_info_count", "line_prefix_count"}),
        "v2 target transforms",
    )
    if transforms["kind"] != "configured_drop/v1":
        _invalid("v2 target transforms are unsupported")
    _digest(transforms["selection_digest"], "target transform digest")
    _nonnegative_int(transforms["fenced_info_count"], "fenced info count", MAX_ROWS)
    _nonnegative_int(transforms["line_prefix_count"], "line prefix count", MAX_ROWS)
    split = _exact(recipe["split"], frozenset({"kind", "seed_sha256", "allocations"}), "v2 split")
    if split["kind"] != "group_hash_rank":
        _invalid("v2 split kind is unsupported")
    _digest(split["seed_sha256"], "split seed digest")
    allocations = split["allocations"]
    if type(allocations) is not list or len(allocations) != 2:
        _invalid("v2 split allocations are invalid")
    checked_names: list[str] = []
    for raw in allocations:
        allocation = _exact(raw, frozenset({"name", "weight"}), "v2 split allocation")
        if type(allocation["name"]) is not str:
            _invalid("v2 split allocation name is invalid")
        checked_names.append(allocation["name"])
        _positive_int(allocation["weight"], "v2 split allocation weight", 2**31 - 1)
    if checked_names != ["train", "validation"]:
        _invalid("v2 split allocations must be train then validation")
    return recipe


def _verify_bytes_v2(
    path: Path,
    manifest_raw: bytes,
    dataset_raw: bytes,
    *,
    require_content_addressed_name: bool = True,
) -> VerifiedPreparedDatasetV1:
    manifest = _exact(
        _parse_canonical_object(manifest_raw, MAX_MANIFEST_BYTES, "manifest"),
        _MANIFEST_FIELDS_V2,
        "v2 manifest",
    )
    if manifest["schema_version"] != ARTIFACT_SCHEMA_VERSION_V2 or manifest["format"] != MESSAGES_FORMAT:
        _invalid("v2 manifest version or format is unsupported")
    if manifest["row_schema_version"] != ROW_SCHEMA_VERSION_V2:
        _invalid("v2 row schema version is unsupported")
    dataset_digest = _digest(manifest["dataset_digest"], "dataset_digest")
    dataset_id = manifest["dataset_id"]
    if (
        type(dataset_id) is not str
        or dataset_id != "dataset-" + dataset_digest
        or (require_content_addressed_name and path.name != dataset_id)
    ):
        _invalid("v2 dataset identity is invalid")
    source = _exact(
        manifest["source"],
        frozenset({"bundle_digest", "structure_set_digest", "item_count"}),
        "v2 source",
    )
    bundle_digest = _digest(source["bundle_digest"], "bundle_digest")
    structure_set_digest = _digest(source["structure_set_digest"], "structure_set_digest")
    item_count = _positive_int(source["item_count"], "item_count", MAX_ROWS * 32)
    projection = _exact(
        manifest["projection"],
        frozenset({"structure_ref", "name", "field_ref"}),
        "v2 projection",
    )
    projection_config = ProjectionV1.from_dict(
        {"structure_ref": projection["structure_ref"], "name": projection["name"]}
    )
    if type(projection["field_ref"]) is not str or not projection["field_ref"]:
        _invalid("v2 projection field_ref is invalid")
    projection_digest = _digest(manifest["projection_digest"], "projection_digest")
    if _domain_digest(_PROJECTION_DOMAIN_V2, projection) != projection_digest:
        _invalid("v2 projection digest is invalid")
    recipe = _validate_recipe_v2(manifest["recipe"])
    if recipe["target_projection"] != projection_config.to_dict():
        _invalid("v2 recipe projection does not match resolved projection")
    prompt_variants = {
        entry["name"]: entry for entry in recipe["context_package"]["prompt_variants"]
    }

    row_count = _positive_int(manifest["row_count"], "row_count", MAX_ROWS)
    dataset_bytes = _positive_int(manifest["dataset_bytes"], "dataset_bytes", MAX_DATASET_BYTES)
    dataset_sha256 = _digest(manifest["dataset_sha256"], "dataset_sha256")
    row_ids_sha256 = _digest(manifest["row_ids_sha256"], "row_ids_sha256")
    group_count = _positive_int(manifest["group_count"], "group_count", MAX_ROWS)
    group_ids_sha256 = _digest(manifest["group_ids_sha256"], "group IDs digest")
    split_counts = _exact(manifest["split_counts"], frozenset({"train", "validation"}), "split_counts")
    checked_counts = {
        name: _positive_int(split_counts[name], "split count", MAX_ROWS)
        for name in ("train", "validation")
    }
    if sum(checked_counts.values()) != row_count:
        _invalid("v2 split counts do not match row_count")
    if len(dataset_raw) != dataset_bytes or hashlib.sha256(dataset_raw).hexdigest() != dataset_sha256:
        _invalid("v2 dataset bytes do not match manifest")
    if not dataset_raw.endswith(b"\n"):
        _invalid("v2 dataset JSONL is not newline terminated")
    lines = dataset_raw[:-1].split(b"\n")
    if len(lines) != row_count or any(not line for line in lines):
        _invalid("v2 dataset row count is invalid")

    raw_lineage = manifest["lineage"]
    if type(raw_lineage) is not list or len(raw_lineage) != row_count:
        _invalid("v2 manifest lineage is invalid")
    lineage: list[dict[str, object]] = []
    for raw in raw_lineage:
        entry = _exact(raw, _LINEAGE_FIELDS_V2, "v2 row lineage")
        for field in (
            "prompt_template_sha256", "separator_sha256", "rendered_user_sha256", "assistant_sha256"
        ):
            _digest(entry[field], field)
        if type(entry["prompt_variant"]) is not str or not entry["prompt_variant"]:
            _invalid("v2 prompt variant lineage is invalid")
        variant = prompt_variants.get(entry["prompt_variant"])
        if (
            variant is None
            or variant["prompt_sha256"] != entry["prompt_template_sha256"]
            or variant["separator_sha256"] != entry["separator_sha256"]
        ):
            _invalid("v2 prompt lineage does not match its configured variant")
        lineage.append(entry)
    lineage_digest = _digest(manifest["lineage_digest"], "lineage_digest")
    if _domain_digest(_LINEAGE_DOMAIN_V2, lineage) != lineage_digest:
        _invalid("v2 lineage digest is invalid")

    row_ids: list[str] = []
    observed_counts = {"train": 0, "validation": 0}
    observed_groups: dict[str, str] = {}
    referenced_items: set[str] = set()
    for index, line in enumerate(lines):
        row = _exact(_parse_canonical_object(line, MAX_DATASET_BYTES, "dataset row"), _ROW_FIELDS_V2, "v2 row")
        if row["schema_version"] != ROW_SCHEMA_VERSION_V2 or row["format"] != MESSAGES_FORMAT:
            _invalid("v2 row version or format is unsupported")
        row_id = row["row_id"]
        target_item_id = row["target_item_id"]
        context_item_ids = row["context_item_ids"]
        group_id = row["group_id"]
        split = row["split"]
        messages = row["messages"]
        if type(row_id) is not str or _ROW_ID.fullmatch(row_id) is None:
            _invalid("v2 row_id is invalid")
        if type(target_item_id) is not str or _ITEM_ID.fullmatch(target_item_id) is None:
            _invalid("v2 target item_id is invalid")
        if type(context_item_ids) is not list or not context_item_ids or len(context_item_ids) != len(set(context_item_ids)):
            _invalid("v2 context item IDs are invalid")
        if target_item_id in context_item_ids or any(type(item) is not str or _ITEM_ID.fullmatch(item) is None for item in context_item_ids):
            _invalid("v2 target/context references are invalid")
        if type(group_id) is not str or not group_id or split not in observed_counts:
            _invalid("v2 group or split is invalid")
        if type(messages) is not list or len(messages) != 2:
            _invalid("v2 rows require exactly two messages")
        expected_roles = ("user", "assistant")
        contents: list[str] = []
        for message, role in zip(messages, expected_roles):
            checked = _exact(message, frozenset({"role", "content"}), "v2 message")
            if checked["role"] != role or type(checked["content"]) is not str or not checked["content"]:
                _invalid("v2 message order or content is invalid")
            contents.append(checked["content"])
        entry = lineage[index]
        if (
            entry["row_id"] != row_id
            or entry["target_item_id"] != target_item_id
            or entry["context_item_ids"] != context_item_ids
            or entry["group_id"] != group_id
            or hashlib.sha256(contents[0].encode("utf-8")).hexdigest() != entry["rendered_user_sha256"]
            or hashlib.sha256(contents[1].encode("utf-8")).hexdigest() != entry["assistant_sha256"]
        ):
            _invalid("v2 row does not match its manifest lineage")
        expected_row_id = "row-" + _domain_digest(
            _ROW_ID_DOMAIN_V2,
            {
                "source_bundle_digest": bundle_digest,
                "target_item_id": target_item_id,
                "context_item_ids": context_item_ids,
                "target_projection": projection,
                "prompt_variant": entry["prompt_variant"],
                "prompt_template_sha256": entry["prompt_template_sha256"],
                "separator_sha256": entry["separator_sha256"],
                "rendered_user_sha256": entry["rendered_user_sha256"],
                "assistant_sha256": entry["assistant_sha256"],
                "group_id": group_id,
                "format": MESSAGES_FORMAT,
            },
        )
        if row_id != expected_row_id:
            _invalid("v2 row_id does not bind row semantics")
        prior_split = observed_groups.setdefault(group_id, split)
        if prior_split != split:
            _invalid("v2 group crosses declared splits")
        row_ids.append(row_id)
        observed_counts[split] += 1
        referenced_items.add(target_item_id)
        referenced_items.update(context_item_ids)
    if len(row_ids) != len(set(row_ids)) or observed_counts != checked_counts:
        _invalid("v2 row identities or split counts are invalid")
    if len(referenced_items) > item_count:
        _invalid("v2 source item count is inconsistent")
    if len(observed_groups) != group_count:
        _invalid("v2 group count is inconsistent")
    if _stream_digest(tuple(sorted(observed_groups))) != group_ids_sha256:
        _invalid("v2 group IDs digest is invalid")
    if _stream_digest(tuple(row_ids)) != row_ids_sha256:
        _invalid("v2 row IDs digest is invalid")
    if recipe["context_package"]["package_count"] != row_count:
        _invalid("v2 recipe package count is inconsistent")

    basis = {
        "source_bundle_digest": bundle_digest,
        "source_structure_set_digest": structure_set_digest,
        "projection": projection,
        "projection_digest": projection_digest,
        "recipe": recipe,
        "lineage": lineage,
        "lineage_digest": lineage_digest,
        "group_count": group_count,
        "group_ids_sha256": group_ids_sha256,
        "row_count": row_count,
        "dataset_bytes": dataset_bytes,
        "dataset_sha256": dataset_sha256,
        "row_ids_sha256": row_ids_sha256,
        "split_counts": checked_counts,
    }
    if _domain_digest(_DATASET_DOMAIN_V2, basis) != dataset_digest:
        _invalid("v2 dataset digest is invalid")
    identity = DatasetSemanticIdentityV1(
        dataset_id,
        dataset_digest,
        row_count,
        dataset_bytes,
        dataset_sha256,
        row_ids_sha256,
        MappingProxyType(checked_counts),
    )
    return VerifiedPreparedDatasetV1(path, identity)


def _verify_bytes(
    path: Path,
    manifest_raw: bytes,
    dataset_raw: bytes,
    *,
    require_content_addressed_name: bool = True,
    expected_artifact_version: str | None = None,
) -> VerifiedPreparedDatasetV1:
    parsed = _parse_canonical_object(manifest_raw, MAX_MANIFEST_BYTES, "manifest")
    version = parsed.get("schema_version")
    if expected_artifact_version is not None and version != expected_artifact_version:
        _invalid("manifest schema version does not match the requested verifier")
    if version == ARTIFACT_SCHEMA_VERSION:
        return _verify_bytes_v1(
            path,
            manifest_raw,
            dataset_raw,
            require_content_addressed_name=require_content_addressed_name,
        )
    if version == ARTIFACT_SCHEMA_VERSION_V2:
        return _verify_bytes_v2(
            path,
            manifest_raw,
            dataset_raw,
            require_content_addressed_name=require_content_addressed_name,
        )
    _invalid("manifest schema version is unsupported")


def verify_prepared_dataset_v1(path: Path) -> VerifiedPreparedDatasetV1:
    """Attest one exact point-in-time observation; no post-return immutability is claimed."""

    if type(path) is not _PATH_TYPE:
        raise TypeError("path must be a Path")
    verified, _dataset_raw = _snapshot(
        path,
        require_content_addressed_name=True,
        expected_artifact_version=ARTIFACT_SCHEMA_VERSION,
    )
    return verified


def snapshot_prepared_dataset_v1(
    path: Path,
) -> tuple[VerifiedPreparedDatasetV1, bytes]:
    """Return the attested identity and the exact JSONL bytes from that attestation."""

    if type(path) is not _PATH_TYPE:
        raise TypeError("path must be a Path")
    return _snapshot(
        path,
        require_content_addressed_name=True,
        expected_artifact_version=ARTIFACT_SCHEMA_VERSION,
    )


def verify_prepared_dataset_v2(path: Path) -> VerifiedPreparedDatasetV1:
    """Attest one strict prepared message-dataset v2 observation."""

    if type(path) is not _PATH_TYPE:
        raise TypeError("path must be a Path")
    verified, _dataset_raw = _snapshot(
        path,
        require_content_addressed_name=True,
        expected_artifact_version=ARTIFACT_SCHEMA_VERSION_V2,
    )
    return verified


def snapshot_prepared_dataset_v2(path: Path) -> tuple[VerifiedPreparedDatasetV1, bytes]:
    """Return a strict v2 attestation and its exact JSONL bytes."""

    if type(path) is not _PATH_TYPE:
        raise TypeError("path must be a Path")
    return _snapshot(
        path,
        require_content_addressed_name=True,
        expected_artifact_version=ARTIFACT_SCHEMA_VERSION_V2,
    )


def _verify(path: Path, *, require_content_addressed_name: bool) -> VerifiedPreparedDatasetV1:
    verified, _dataset_raw = _snapshot(
        path, require_content_addressed_name=require_content_addressed_name
    )
    return verified


def _snapshot(
    path: Path,
    *,
    require_content_addressed_name: bool,
    expected_artifact_version: str | None = None,
) -> tuple[VerifiedPreparedDatasetV1, bytes]:
    authority = _RetainedDatasetDirectory(path)
    retained: list[_RetainedMember] = []
    verified: VerifiedPreparedDatasetV1 | None = None
    dataset_raw: bytes | None = None
    close_failed = False
    try:
        manifest_member = _RetainedMember(
            path / "manifest.json",
            MAX_MANIFEST_BYTES,
            authority.member_identities["manifest.json"],
        )
        retained.append(manifest_member)
        dataset_member = _RetainedMember(
            path / "dataset.jsonl",
            MAX_DATASET_BYTES,
            authority.member_identities["dataset.jsonl"],
        )
        retained.append(dataset_member)
        manifest_raw = manifest_member.read()
        dataset_raw = dataset_member.read()
        verified = _verify_bytes(
            path,
            manifest_raw,
            dataset_raw,
            require_content_addressed_name=require_content_addressed_name,
            expected_artifact_version=expected_artifact_version,
        )
        manifest_member.verify_final_identity()
        dataset_member.verify_final_identity()
        authority.verify_final_inventory()
    finally:
        for member in reversed(retained):
            if not member.close():
                close_failed = True
    if close_failed:
        _invalid("prepared dataset member could not be closed safely")
    if verified is None:
        _invalid("prepared dataset attestation did not complete")
    if dataset_raw is None:
        _invalid("prepared dataset snapshot did not complete")
    return verified, dataset_raw


def _write_member(path: Path, payload: bytes) -> None:
    descriptor: int | None = None
    failed = False
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
        os.chmod(path, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        failed = True
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                failed = True
    if failed:
        raise RuntimeError("prepared dataset member could not be written") from None


def _write_stage_member(stage: _OwnedStageAuthority, name: str, payload: bytes) -> None:
    stage.verify_binding()
    if stage.descriptor is None:
        _write_member(stage.path / name, payload)
        stage.verify_binding()
        return
    descriptor: int | None = None
    failed = False
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=stage.descriptor,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        failed = True
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                failed = True
    if failed:
        raise RuntimeError("prepared dataset member could not be written") from None
    stage.verify_binding()


def _relative_inventory(stage: _OwnedStageAuthority) -> dict[str, tuple[int, int, int, int, int]]:
    stage.verify_binding()
    if stage.descriptor is None:
        return _inventory(stage.path, stage.identity)
    result: dict[str, tuple[int, int, int, int, int]] = {}
    try:
        for name in os.listdir(stage.descriptor):
            if name not in _EXPECTED_INVENTORY or name in result:
                _invalid("prepared dataset inventory is invalid")
            info = os.stat(name, dir_fd=stage.descriptor, follow_symlinks=False)
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_ISLNK(info.st_mode)
                or info.st_nlink != 1
            ):
                _invalid("prepared dataset member is not a plain file")
            result[name] = _identity(info)
    except DatasetPrepValidationError:
        raise
    except BaseException:
        _invalid("prepared dataset inventory is unavailable")
    if frozenset(result) != _EXPECTED_INVENTORY:
        _invalid("prepared dataset inventory is invalid")
    stage.verify_binding()
    return result


def _verify_owned_stage(stage: _OwnedStageAuthority) -> VerifiedPreparedDatasetV1:
    if stage.descriptor is None:
        return _verify(stage.path, require_content_addressed_name=False)
    inventory = _relative_inventory(stage)
    retained: list[_RetainedRelativeMember] = []
    verified: VerifiedPreparedDatasetV1 | None = None
    close_failed = False
    try:
        manifest = _RetainedRelativeMember(
            stage.descriptor, "manifest.json", MAX_MANIFEST_BYTES, inventory["manifest.json"]
        )
        retained.append(manifest)
        dataset = _RetainedRelativeMember(
            stage.descriptor, "dataset.jsonl", MAX_DATASET_BYTES, inventory["dataset.jsonl"]
        )
        retained.append(dataset)
        verified = _verify_bytes(
            stage.path,
            manifest.read(),
            dataset.read(),
            require_content_addressed_name=False,
        )
        manifest.verify_final_identity()
        dataset.verify_final_identity()
        if _relative_inventory(stage) != inventory:
            _invalid("prepared dataset inventory changed during verification")
    finally:
        for member in reversed(retained):
            if not member.close():
                close_failed = True
    if close_failed:
        _invalid("prepared dataset member could not be closed safely")
    if verified is None:
        _invalid("prepared dataset attestation did not complete")
    return verified


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        raise OSError("directory durability synchronization is unavailable")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_directory_authority(descriptor: int | None, path: Path) -> None:
    if descriptor is not None:
        os.fsync(descriptor)
    else:
        _fsync_directory(path)


def _rename_noreplace(source: Path, destination: Path) -> None:
    if source.parent != destination.parent:
        raise RuntimeError("dataset publication crossed directories")
    if os.name == "nt":
        os.rename(source, destination)
        return
    if sys.platform != "linux":
        raise RuntimeError("atomic no-replace publication is unavailable")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace publication is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), str(destination))


def _rename_noreplace_relative(directory_fd: int, source: str, destination: str) -> None:
    if sys.platform != "linux":
        raise RuntimeError("descriptor-relative no-replace publication is unavailable")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("descriptor-relative no-replace publication is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(directory_fd, os.fsencode(source), directory_fd, os.fsencode(destination), 1) != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))


def _destination_matches_stage(stage: _OwnedStageAuthority, destination: Path) -> bool:
    try:
        if stage.root.descriptor is not None:
            declared = os.stat(destination.name, dir_fd=stage.root.descriptor, follow_symlinks=False)
            return stat.S_ISDIR(declared.st_mode) and _same_directory(_identity(declared), stage.identity)
        return _same_directory(_plain_directory(destination), stage.identity)
    except BaseException:
        return False


def _scrub_retained_stage_members(stage: _OwnedStageAuthority) -> bool:
    """Best-effort logical byte scrub through retained authority after a bad rename."""

    if stage.descriptor is None or stage.closed:
        return False
    failed = False
    try:
        retained = _identity(os.fstat(stage.descriptor))
        if not _same_directory(retained, stage.identity):
            _invalid("published dataset retained authority changed")
        names = os.listdir(stage.descriptor)
        if frozenset(names) != _EXPECTED_INVENTORY or len(names) != len(_EXPECTED_INVENTORY):
            _invalid("published dataset scrub inventory is invalid")
        for name in sorted(names):
            declared = os.stat(name, dir_fd=stage.descriptor, follow_symlinks=False)
            if not stat.S_ISREG(declared.st_mode) or stat.S_ISLNK(declared.st_mode):
                _invalid("published dataset scrub member is invalid")
            expected = _identity(declared)
            os.chmod(name, 0o600, dir_fd=stage.descriptor, follow_symlinks=False)
            descriptor = os.open(
                name,
                os.O_WRONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=stage.descriptor,
            )
            try:
                if not _same_file(_identity(os.fstat(descriptor)), expected):
                    _invalid("published dataset scrub member identity changed")
                os.ftruncate(descriptor, 0)
                os.fsync(descriptor)
                if os.fstat(descriptor).st_size != 0:
                    _invalid("published dataset member bytes could not be scrubbed")
            finally:
                os.close(descriptor)
        os.fsync(stage.descriptor)
    except BaseException:
        failed = True
    return not failed


def _commit_owned_stage(stage: _OwnedStageAuthority, destination: Path) -> None:
    stage.verify_binding()
    if stage.descriptor is not None:
        assert stage.root.descriptor is not None
        _rename_noreplace_relative(stage.root.descriptor, stage.name, destination.name)
    elif stage.native_handle is not None and stage.root.native_handle is not None:
        _win_rename_directory_relative(
            stage.native_handle, 0, str(destination)
        )
    else:
        _rename_noreplace(stage.path, destination)
    stage.published_name = destination.name
    if not _destination_matches_stage(stage, destination):
        raise _PostRenameIdentityMismatch(_scrub_retained_stage_members(stage)) from None
    stage.committed = True


def _owned_stage_name_absent(stage: _OwnedStageAuthority) -> bool:
    try:
        if stage.root.descriptor is not None:
            os.stat(stage.name, dir_fd=stage.root.descriptor, follow_symlinks=False)
        else:
            stage.path.lstat()
    except FileNotFoundError:
        return True
    except BaseException:
        return False
    return False


def _cleanup_owned_stage(stage: _OwnedStageAuthority, semantic_identity: DatasetSemanticIdentityV1) -> None:
    """Remove only the retained, still-bound stage; never follow a substituted name."""

    if stage.committed or stage.published_name is not None:
        return
    failed = False
    try:
        stage.verify_binding()
        if stage.descriptor is not None:
            assert stage.root.descriptor is not None
            inventory: dict[str, tuple[int, int, int, int, int]] = {}
            for name in os.listdir(stage.descriptor):
                if name not in _EXPECTED_INVENTORY or name in inventory:
                    _invalid("owned stage inventory is invalid")
                info = os.stat(name, dir_fd=stage.descriptor, follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                    _invalid("owned stage member is not a plain file")
                inventory[name] = _identity(info)
            stage.verify_binding()
            for name in sorted(inventory):
                descriptor = os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=stage.descriptor,
                )
                try:
                    opened = _identity(os.fstat(descriptor))
                    if opened != inventory[name]:
                        _invalid("owned stage member identity changed")
                    claim = f".cleanup-{secrets.token_hex(16)}"
                    _rename_noreplace_relative(stage.descriptor, name, claim)
                    claimed = _identity(os.stat(claim, dir_fd=stage.descriptor, follow_symlinks=False))
                    if claimed != opened:
                        _invalid("owned stage cleanup claim changed")
                finally:
                    os.close(descriptor)
                if _identity(os.stat(claim, dir_fd=stage.descriptor, follow_symlinks=False)) != opened:
                    _invalid("owned stage cleanup claim changed")
                os.unlink(claim, dir_fd=stage.descriptor)
            if os.listdir(stage.descriptor):
                _invalid("owned stage cleanup inventory is not empty")
            os.fsync(stage.descriptor)
            stage.verify_binding()
            os.rmdir(stage.name, dir_fd=stage.root.descriptor)
        else:
            inventory: dict[str, tuple[int, int, int, int, int]] = {}
            stage.verify_binding()
            with os.scandir(stage.path) as entries:
                for entry in entries:
                    if entry.name not in _EXPECTED_INVENTORY or entry.name in inventory:
                        _invalid("owned stage inventory is invalid")
                    info = (stage.path / entry.name).lstat()
                    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or _is_reparse(info):
                        _invalid("owned stage member is not a plain file")
                    inventory[entry.name] = _identity(info)
            stage.verify_binding()
            for name, expected in inventory.items():
                member = stage.path / name
                if _identity(member.lstat()) != expected:
                    _invalid("owned stage member identity changed")
                try:
                    os.chmod(member, 0o600)
                except OSError:
                    pass
                if not _same_file(_identity(member.lstat()), expected):
                    _invalid("owned stage member identity changed")
                if stage.native_handle is not None:
                    _win_delete_member_exact(member, expected)
                else:
                    member.unlink()
            stage.verify_binding()
            if stage.native_handle is not None:
                _win_mark_directory_delete(stage.native_handle)
                if not stage.close():
                    _invalid("owned stage handle could not be closed safely")
            else:
                stage.close()
                stage.path.rmdir()
        _sync_directory_authority(stage.root.descriptor, stage.root.path)
        if not _owned_stage_name_absent(stage):
            _invalid("owned stage name removal could not be verified")
    except BaseException:
        failed = True
    if failed:
        raise DatasetPublicationUncertainV1(
            semantic_identity,
            DatasetPublicationUncertaintyPhaseV1.STAGE_CLEANUP,
        ) from None
    stage.cleaned = True


def retry_dataset_root_durability_v1(output_root: Path) -> None:
    failed = type(output_root) is not _PATH_TYPE
    if not failed:
        try:
            _plain_directory(output_root)
            _fsync_directory(output_root)
        except BaseException:
            failed = True
    if failed:
        raise DatasetPrepDurabilityError() from None


def _publish(output_root: Path, prepared) -> VerifiedPreparedDatasetV1:
    if type(output_root) is not _PATH_TYPE:
        raise TypeError("output_root must be a Path")
    root: _OutputRootAuthority | None = None
    coordination: _OutputRootCoordination | None = None
    stage: _OwnedStageAuthority | None = None
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        root = _OutputRootAuthority(output_root)
        coordination = _OutputRootCoordination(root)
        destination = output_root / prepared.identity.dataset_id
        try:
            destination.lstat()
        except FileNotFoundError:
            pass
        else:
            try:
                existing = _verify(destination, require_content_addressed_name=True)
            except DatasetPrepValidationError:
                raise DatasetPrepCollisionError("prepared dataset destination collision") from None
            if existing.semantic_identity != prepared.identity:
                raise DatasetPrepCollisionError("prepared dataset destination collision") from None
            coordination.close()
            root.close()
            return existing
        stage = _OwnedStageAuthority.create(root, prepared.identity.dataset_id)
    except (DatasetPrepValidationError, DatasetPrepCollisionError, TypeError):
        if coordination is not None:
            coordination.close()
        if root is not None:
            root.close()
        raise
    except BaseException:
        if coordination is not None:
            coordination.close()
        if root is not None:
            root.close()
        _invalid("prepared dataset staging directory could not be created")
    try:
        assert root is not None and stage is not None
        _write_stage_member(stage, "dataset.jsonl", prepared.dataset_raw)
        _write_stage_member(stage, "manifest.json", prepared.manifest_raw)
        for name in _EXPECTED_INVENTORY:
            try:
                if stage.descriptor is not None:
                    os.chmod(name, 0o400, dir_fd=stage.descriptor, follow_symlinks=False)
                else:
                    os.chmod(stage.path / name, stat.S_IREAD)
            except OSError:
                pass
        directory_durability_unproven = False
        try:
            _sync_directory_authority(stage.descriptor, stage.path)
        except BaseException:
            directory_durability_unproven = True
        staged = _verify_owned_stage(stage)
        if staged.semantic_identity != prepared.identity:
            _invalid("staged dataset identity is inconsistent")
        post_rename_phase: DatasetPublicationUncertaintyPhaseV1 | None = None
        commit_collision = False
        try:
            _commit_owned_stage(stage, destination)
        except _PostRenameIdentityMismatch as mismatch:
            post_rename_phase = (
                DatasetPublicationUncertaintyPhaseV1.FINAL_VERIFICATION
                if mismatch.scrubbed
                else DatasetPublicationUncertaintyPhaseV1.STAGE_CLEANUP
            )
        except OSError as exc:
            if exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
                commit_collision = True
            else:
                raise
        if post_rename_phase is not None:
            raise DatasetPublicationUncertainV1(prepared.identity, post_rename_phase) from None
        if commit_collision:
            winner: VerifiedPreparedDatasetV1 | None = None
            try:
                candidate = _verify(destination, require_content_addressed_name=True)
                if candidate.semantic_identity == prepared.identity:
                    winner = candidate
            except BaseException:
                pass
            _cleanup_owned_stage(stage, prepared.identity)
            if winner is None:
                raise DatasetPrepCollisionError("prepared dataset destination collision") from None
            return winner
        parent_uncertain = directory_durability_unproven
        try:
            _sync_directory_authority(root.descriptor, output_root)
        except BaseException:
            parent_uncertain = True
        if parent_uncertain:
            raise DatasetPublicationUncertainV1(
                prepared.identity, DatasetPublicationUncertaintyPhaseV1.PARENT_DURABILITY
            ) from None
        verified = None
        final_uncertain = False
        try:
            verified = _verify(destination, require_content_addressed_name=True)
        except BaseException:
            final_uncertain = True
        if final_uncertain:
            raise DatasetPublicationUncertainV1(
                prepared.identity, DatasetPublicationUncertaintyPhaseV1.FINAL_VERIFICATION
            ) from None
        if verified.semantic_identity != prepared.identity:
            raise DatasetPublicationUncertainV1(
                prepared.identity, DatasetPublicationUncertaintyPhaseV1.FINAL_VERIFICATION
            ) from None
        return verified
    except DatasetPublicationUncertainV1:
        raise
    except (DatasetPrepValidationError, DatasetPrepCollisionError, TypeError):
        if stage is not None and not stage.committed and not stage.cleaned:
            _cleanup_owned_stage(stage, prepared.identity)
        raise
    except BaseException:
        if stage is not None and not stage.committed and not stage.cleaned:
            _cleanup_owned_stage(stage, prepared.identity)
        raise RuntimeError("prepared dataset could not be published") from None
    finally:
        if stage is not None:
            stage.close()
        if coordination is not None:
            coordination.close()
        if root is not None:
            root.close()


def prepare_dataset_v1(config: DatasetPrepConfigV1, output_root: Path) -> VerifiedPreparedDatasetV1:
    """Verify, prepare, and privately publish under the cooperative trust boundary.

    Publishers using this API serialize on an authenticated output-root lock.
    Code running as the same OS identity must be trusted not to subvert retained
    handles or mutate the root outside this protocol.
    """

    if type(config) is not DatasetPrepConfigV1:
        raise TypeError("config must be exact DatasetPrepConfigV1")
    bundle = load_verified_normalized_bundle_v1(config.source_bundle_path)
    prepared = build_prepared_dataset_v1(bundle, config)
    uncertainty: tuple[DatasetSemanticIdentityV1, DatasetPublicationUncertaintyPhaseV1] | None = None
    try:
        return _publish(output_root, prepared)
    except DatasetPublicationUncertainV1 as error:
        uncertainty = (error.semantic_identity, error.phase)
    assert uncertainty is not None
    raise DatasetPublicationUncertainV1(*uncertainty) from None


def prepare_dataset_v2(config: DatasetPrepConfigV2, output_root: Path) -> VerifiedPreparedDatasetV1:
    """Verify, prepare, and publish an additive context-package message dataset."""

    if type(config) is not DatasetPrepConfigV2:
        raise TypeError("config must be exact DatasetPrepConfigV2")
    bundle = load_verified_normalized_bundle_v1(config.source_bundle_path)
    prepared = build_prepared_dataset_v2(bundle, config)
    uncertainty: tuple[DatasetSemanticIdentityV1, DatasetPublicationUncertaintyPhaseV1] | None = None
    try:
        return _publish(output_root, prepared)
    except DatasetPublicationUncertainV1 as error:
        uncertainty = (error.semantic_identity, error.phase)
    assert uncertainty is not None
    raise DatasetPublicationUncertainV1(*uncertainty) from None
