"""Import-light, identity-bound credential file handling for protected HF routes."""

from __future__ import annotations

import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

from tuner.core.exceptions import CloudProviderError


_MAX_SECRET_FILE_BYTES = 64 * 1024
_DECLARATION_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class HFSecretFileClaim:
    """Metadata-only selection of a credential file.

    The selection deliberately stores neither file identity nor content.  The
    durable provider claim must be won before ``read_claimed_hf_token`` performs
    the single complete open/read and authenticates the file for that attempt.
    """

    root: Path
    path: Path


def preflight_hf_secret_file(value: object, *, context) -> HFSecretFileClaim:
    """Bind one explicit regular link-free file without reading its bytes."""

    if "HF_TOKEN" in os.environ or "HF_API_KEY" in os.environ:
        raise CloudProviderError(
            "Protected HF routes reject ambient Hugging Face credentials; "
            "authority must come only from the explicit file."
        )
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise CloudProviderError("Protected HF routes require an explicit --env-file selection.")
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raw = context.invocation_cwd / raw
    path = Path(os.path.abspath(raw))
    resolved_roots: list[Path] = []
    for selected_root in (context.project_root, context.config_root):
        try:
            candidate = Path(selected_root).resolve(strict=True)
        except OSError:
            continue
        if candidate not in resolved_roots:
            resolved_roots.append(candidate)
    roots = tuple(resolved_roots)
    containing = [root for root in roots if _is_relative_to(path, root)]
    if not containing:
        raise CloudProviderError(
            "Protected HF credential selection must remain within the project/config boundary."
        )
    root = max(containing, key=lambda candidate: len(candidate.parts))
    _assert_link_free_chain(root, path)
    try:
        info = path.lstat()
    except OSError:
        raise CloudProviderError("Protected HF credential file is unavailable.") from None
    if info.st_size <= 0 or info.st_size > _MAX_SECRET_FILE_BYTES:
        raise CloudProviderError("Protected HF credential file must be non-empty and bounded.")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise CloudProviderError(
            "Protected HF credential path cannot traverse links or reparse points."
        )
    return HFSecretFileClaim(root=root, path=path)


def read_claimed_hf_token(claim: HFSecretFileClaim) -> str:
    """Safely open/read the selected file once after durable provider claim."""

    if "HF_TOKEN" in os.environ or "HF_API_KEY" in os.environ:
        raise CloudProviderError(
            "Protected HF routes reject ambient Hugging Face credentials; "
            "authority must come only from the explicit file."
        )

    try:
        raw = (
            _read_windows_snapshot(claim)
            if os.name == "nt"
            else _read_posix_snapshot(claim)
        )
    except CloudProviderError:
        raise
    except OSError:
        raise CloudProviderError("Protected HF credential file could not be read safely.") from None
    if len(raw) > _MAX_SECRET_FILE_BYTES:
        raise CloudProviderError("Protected HF credential file exceeds its bound.")
    try:
        document = raw.decode("utf-8")
    except UnicodeError:
        raise CloudProviderError("Protected HF credential file must be valid UTF-8.") from None
    return _parse_strict_hf_token(document)


def _read_posix_snapshot(claim: HFSecretFileClaim) -> bytes:
    """Read through one held, handle-relative, no-follow ancestor chain.

    The guarantee is deliberately scoped to the opened object and held parent
    handles.  A concurrent rename after the final open cannot change the bytes
    read from that descriptor, but this does not claim the pathname still names
    the same object after the read.
    """

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise CloudProviderError("Protected HF credential safety primitives are unavailable.")
    relative = claim.path.relative_to(claim.root)
    handles: list[int] = []
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        current = os.open(claim.root, directory_flags)
        handles.append(current)
        parts = relative.parts
        if not parts:
            raise CloudProviderError("Protected HF credential path must identify a file.")
        for part in parts[:-1]:
            current = os.open(part, directory_flags, dir_fd=current)
            handles.append(current)
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=current,
        )
        handles.append(descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise CloudProviderError("Protected HF credential path must identify a regular file.")
        raw = _read_fd_bounded(descriptor)
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            raise CloudProviderError("Protected HF credential file changed during authorization.")
        return raw
    finally:
        for descriptor in reversed(handles):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _read_fd_bounded(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    remaining = _MAX_SECRET_FILE_BYTES + 1
    while remaining:
        chunk = os.read(descriptor, min(65536, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_windows_snapshot(claim: HFSecretFileClaim) -> bytes:
    """Read while holding native handles that deny rename/write/delete races."""

    if sys.platform != "win32":
        raise CloudProviderError("Protected HF Windows credential safety is unavailable.")
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        raise CloudProviderError("Protected HF Windows credential safety is unavailable.") from None

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    kernel32.GetFileInformationByHandleEx.argtypes = (
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
    )
    kernel32.GetFileInformationByHandleEx.restype = wintypes.BOOL
    kernel32.GetFinalPathNameByHandleW.argtypes = (
        wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD,
    )
    kernel32.GetFinalPathNameByHandleW.restype = wintypes.DWORD
    kernel32.GetFileInformationByHandle.argtypes = (wintypes.HANDLE, wintypes.LPVOID)
    kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
    kernel32.ReadFile.argtypes = (
        wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID,
    )
    kernel32.ReadFile.restype = wintypes.BOOL
    invalid = wintypes.HANDLE(-1).value
    share_read = 0x00000001
    open_existing = 3
    backup_semantics = 0x02000000
    open_reparse = 0x00200000
    generic_read = 0x80000000
    handles: list[object] = []

    def opened(path: Path, *, directory: bool) -> object:
        flags = open_reparse | (backup_semantics if directory else 0)
        handle = create_file(str(path), 0 if directory else generic_read, share_read, None, open_existing, flags, None)
        if handle == invalid:
            raise CloudProviderError("Protected HF credential file could not be opened safely.")
        handles.append(handle)
        return handle

    try:
        current = claim.root
        ancestors = [claim.root]
        for part in claim.path.relative_to(claim.root).parts[:-1]:
            current = current / part
            ancestors.append(current)
        root_handle = None
        for ancestor in ancestors:
            handle = opened(ancestor, directory=True)
            root_handle = root_handle or handle
            _reject_windows_reparse(kernel32, handle, expect_directory=True)
        file_handle = opened(claim.path, directory=False)
        _reject_windows_reparse(kernel32, file_handle, expect_directory=False)
        assert root_handle is not None
        root_final = _windows_final_path(kernel32, root_handle)
        file_final = _windows_final_path(kernel32, file_handle)
        prefix = root_final.rstrip("\\/") + "\\"
        if not file_final.casefold().startswith(prefix.casefold()):
            raise CloudProviderError("Protected HF credential handle escapes its allowed root.")
        identity_before = _windows_file_identity(kernel32, file_handle)
        size = identity_before[3]
        if size <= 0 or size > _MAX_SECRET_FILE_BYTES:
            raise CloudProviderError("Protected HF credential file must be non-empty and bounded.")
        raw = _windows_read_all(kernel32, file_handle, size)
        if _windows_file_identity(kernel32, file_handle) != identity_before:
            raise CloudProviderError("Protected HF credential file changed during authorization.")
        return raw
    finally:
        for handle in reversed(handles):
            close_handle(handle)


def _reject_windows_reparse(kernel32, handle, *, expect_directory: bool) -> None:
    import ctypes
    from ctypes import wintypes

    class FILE_ATTRIBUTE_TAG_INFO(ctypes.Structure):
        _fields_ = [("FileAttributes", wintypes.DWORD), ("ReparseTag", wintypes.DWORD)]

    info = FILE_ATTRIBUTE_TAG_INFO()
    if not kernel32.GetFileInformationByHandleEx(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
        raise CloudProviderError("Protected HF credential handle metadata is unavailable.")
    if info.FileAttributes & 0x400:
        raise CloudProviderError("Protected HF credential path cannot traverse reparse points.")
    is_directory = bool(info.FileAttributes & 0x10)
    if is_directory != expect_directory:
        raise CloudProviderError("Protected HF credential handle has an invalid file type.")


def _windows_final_path(kernel32, handle) -> str:
    import ctypes

    required = kernel32.GetFinalPathNameByHandleW(handle, None, 0, 0)
    if not required:
        raise CloudProviderError("Protected HF credential final handle path is unavailable.")
    buffer = ctypes.create_unicode_buffer(required + 1)
    if not kernel32.GetFinalPathNameByHandleW(handle, buffer, len(buffer), 0):
        raise CloudProviderError("Protected HF credential final handle path is unavailable.")
    return buffer.value


def _windows_file_identity(kernel32, handle) -> tuple[int, int, int, int]:
    import ctypes
    from ctypes import wintypes

    class BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD), ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME), ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD), ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD), ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD), ("nFileIndexLow", wintypes.DWORD),
        ]

    info = BY_HANDLE_FILE_INFORMATION()
    if not kernel32.GetFileInformationByHandle(handle, ctypes.byref(info)):
        raise CloudProviderError("Protected HF credential file identity is unavailable.")
    index = (info.nFileIndexHigh << 32) | info.nFileIndexLow
    size = (info.nFileSizeHigh << 32) | info.nFileSizeLow
    modified = (info.ftLastWriteTime.dwHighDateTime << 32) | info.ftLastWriteTime.dwLowDateTime
    return info.dwVolumeSerialNumber, index, modified, size


def _windows_read_all(kernel32, handle, expected_size: int) -> bytes:
    import ctypes
    from ctypes import wintypes

    chunks: list[bytes] = []
    remaining = expected_size
    while remaining:
        size = min(65536, remaining)
        buffer = ctypes.create_string_buffer(size)
        count = wintypes.DWORD()
        if not kernel32.ReadFile(handle, buffer, size, ctypes.byref(count), None):
            raise CloudProviderError("Protected HF credential file could not be read safely.")
        if count.value == 0:
            break
        chunks.append(buffer.raw[: count.value])
        remaining -= count.value
    raw = b"".join(chunks)
    if len(raw) != expected_size:
        raise CloudProviderError("Protected HF credential file changed during authorization.")
    return raw


def _parse_strict_hf_token(document: str) -> str:
    if "\x00" in document:
        raise CloudProviderError("Protected HF credential file contains invalid dotenv syntax.")
    token: str | None = None
    for line in document.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _DECLARATION_RE.fullmatch(line)
        if match is None:
            raise CloudProviderError("Protected HF credential file contains invalid dotenv syntax.")
        key, encoded = match.groups()
        if key != "HF_TOKEN":
            raise CloudProviderError("Protected HF credential file may declare only HF_TOKEN.")
        if token is not None:
            raise CloudProviderError("Protected HF credential file declares HF_TOKEN more than once.")
        token = _decode_token_value(encoded)
    if token is None or not token:
        raise CloudProviderError(
            "Protected HF credential file must declare exactly one non-empty HF_TOKEN."
        )
    return token


def _decode_token_value(encoded: str) -> str:
    value = encoded.strip()
    if not value:
        raise CloudProviderError("Protected HF credential file must declare a non-empty HF_TOKEN.")
    if value[0] in {"'", '"'}:
        quote = value[0]
        if len(value) < 2 or value[-1] != quote or quote in value[1:-1] or "\\" in value:
            raise CloudProviderError("Protected HF credential file contains invalid quoted syntax.")
        value = value[1:-1]
    elif any(character.isspace() or character in "#'\"\\" for character in value):
        raise CloudProviderError("Protected HF credential file contains invalid unquoted syntax.")
    if not value or not _TOKEN_RE.fullmatch(value):
        raise CloudProviderError("Protected HF credential file must declare a non-empty HF_TOKEN.")
    return value


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_mode)


def _assert_link_free_chain(root: Path, path: Path) -> None:
    current = root
    items = [root]
    for part in path.relative_to(root).parts:
        current = current / part
        items.append(current)
    for index, item in enumerate(items):
        try:
            info = item.lstat()
        except OSError:
            raise CloudProviderError("Protected HF credential path is unavailable.") from None
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise CloudProviderError(
                "Protected HF credential path cannot traverse links or reparse points."
            )
        final = index == len(items) - 1
        if final and not stat.S_ISREG(info.st_mode):
            raise CloudProviderError("Protected HF credential path must identify a regular file.")
        if not final and not stat.S_ISDIR(info.st_mode):
            raise CloudProviderError("Protected HF credential path has an invalid parent chain.")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


__all__ = ["HFSecretFileClaim", "preflight_hf_secret_file", "read_claimed_hf_token"]
