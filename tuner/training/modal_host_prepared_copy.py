"""Copy a verified repo publication into a Linux-private preparation root."""

from __future__ import annotations

import os
from pathlib import Path
import stat

from tuner.dataset_prep.publication import (
    ARTIFACT_SCHEMA_VERSION_V2, _verify_bytes, snapshot_prepared_dataset_v2,
)
from tuner.training.input_preparation import PosixPrivatePreparedRootAuthorityV1


class ModalPreparedCopyUnavailable(RuntimeError):
    """Fixed closed diagnostic; no provider or private path text."""


def inspect_mounted_prepared_dataset(source: Path):
    """Read-only byte validation, without asserting private-root admission."""
    if os.name != "posix" or type(source) is not type(Path()) or not source.is_absolute():
        raise ModalPreparedCopyUnavailable("modal_source_inspection_unavailable")
    directory = source.parent if source.name == "dataset.jsonl" else source
    fd = -1
    try:
        info = directory.lstat()
        if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or directory.resolve(strict=True) != directory):
            raise ValueError
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        opened = os.fstat(fd)
        if ((opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino)
                or set(os.listdir(fd)) != {"manifest.json", "dataset.jsonl"}):
            raise ValueError
        manifest = _read_member(fd, "manifest.json", 1024 * 1024)
        dataset = _read_member(fd, "dataset.jsonl", 256 * 1024 * 1024)
        verified = _verify_bytes(
            directory, manifest, dataset, require_content_addressed_name=True,
            expected_artifact_version=ARTIFACT_SCHEMA_VERSION_V2,
        )
        if (_read_member(fd, "manifest.json", 1024 * 1024) != manifest
                or _read_member(fd, "dataset.jsonl", 256 * 1024 * 1024) != dataset
                or os.fstat(fd).st_ino != opened.st_ino):
            raise ValueError
        return verified
    except Exception:
        raise ModalPreparedCopyUnavailable("modal_source_inspection_unavailable") from None
    finally:
        if fd >= 0:
            os.close(fd)


def _read_member(directory: int, name: str, maximum: int) -> bytes:
    member = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    try:
        info = os.fstat(member)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_size < 1 or info.st_size > maximum):
            raise ValueError
        with os.fdopen(os.dup(member), "rb") as handle:
            data = handle.read(maximum + 1)
        if len(data) != info.st_size:
            raise ValueError
        return data
    finally:
        os.close(member)


def stage_published_modal_dataset(
    source: Path, *, private_root: Path,
    expected_dataset_digest: str, expected_content_digest: str | None = None,
) -> Path:
    """Create one exclusive private copy, independently verified on both sides.

    This is the WSL bridge from a Windows-mounted publication; its destination
    is an owner-only native Linux directory accepted by public preparation.
    """
    if os.name != "posix":
        raise ModalPreparedCopyUnavailable("modal_private_preparation_posix_required")
    if (type(source) is not type(Path()) or not source.is_absolute()
            or type(private_root) is not type(Path()) or not private_root.is_absolute()):
        raise ModalPreparedCopyUnavailable("modal_private_preparation_invalid")
    source_dir = source.parent if source.name == "dataset.jsonl" else source
    root_fd = -1
    source_fd = -1
    destination_fd = -1
    try:
        authority = PosixPrivatePreparedRootAuthorityV1()
        attestation = authority.attest(private_root)
        source_info = source_dir.lstat()
        if (not stat.S_ISDIR(source_info.st_mode) or stat.S_ISLNK(source_info.st_mode)
                or source_dir.resolve(strict=True) != source_dir):
            raise ValueError
        source_fd = os.open(source_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        opened = os.fstat(source_fd)
        if (opened.st_dev, opened.st_ino) != (source_info.st_dev, source_info.st_ino):
            raise ValueError
        manifest_data = _read_member(source_fd, "manifest.json", 1024 * 1024)
        copied_data = _read_member(source_fd, "dataset.jsonl", 256 * 1024 * 1024)
        if (_read_member(source_fd, "manifest.json", 1024 * 1024) != manifest_data
                or _read_member(source_fd, "dataset.jsonl", 256 * 1024 * 1024) != copied_data):
            raise ValueError
        root_fd = os.open(private_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        name = f"dataset-{expected_dataset_digest}"
        os.mkdir(name, mode=0o700, dir_fd=root_fd)
        destination_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                 dir_fd=root_fd)
        for member_name, data in (("manifest.json", manifest_data),
                                  ("dataset.jsonl", copied_data)):
            member = os.open(member_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=destination_fd)
            with os.fdopen(member, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        os.fsync(destination_fd)
        copied = private_root / name
        result, copied_snapshot = snapshot_prepared_dataset_v2(copied)
        identity = result.semantic_identity
        if (identity.dataset_digest != expected_dataset_digest
                or (expected_content_digest is not None
                    and identity.dataset_sha256 != expected_content_digest)
                or copied_snapshot != copied_data):
            raise ValueError
        authority.verify(attestation)
        return copied / "dataset.jsonl"
    except Exception:
        raise ModalPreparedCopyUnavailable("modal_private_preparation_failed") from None
    finally:
        for fd in (destination_fd, root_fd, source_fd):
            if fd >= 0:
                os.close(fd)
