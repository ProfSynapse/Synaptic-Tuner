"""Copy a verified repo publication into a Linux-private preparation root."""

from __future__ import annotations

import os
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass
import stat

from tuner.dataset_prep.publication import (
    ARTIFACT_SCHEMA_VERSION_V2, _verify_bytes,
)
from synaptic_tuner.api.v1.training_sources import LocalTrainingInputPathV1, PreparedTrainingInputIdentity
from tuner.training.input_preparation import (
    PosixPrivatePreparedRootAuthorityV1, DatasetPreparedInputFormatVerifierV1,
    NormalizedTrainingInputV1,
)


class ModalPreparedCopyUnavailable(RuntimeError):
    """Fixed closed diagnostic; no provider or private path text."""


def _inventory(directory):
    names = set()
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.name not in {"manifest.json", "dataset.jsonl"} or entry.name in names:
                raise ValueError
            names.add(entry.name)
    if names != {"manifest.json", "dataset.jsonl"}:
        raise ValueError


def _named_directory(path, opened):
    named = path.lstat()
    if (not stat.S_ISDIR(named.st_mode) or stat.S_ISLNK(named.st_mode)
            or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
            or path.resolve(strict=True) != path):
        raise ValueError


def _format_ports(prepared_format):
    if prepared_format == "syntunia-sft-row/v2":
        def parser(path, manifest, raw):
            return _verify_bytes(path, manifest, raw, require_content_addressed_name=True,
                                 expected_artifact_version=ARTIFACT_SCHEMA_VERSION_V2)
        return parser, 1024 * 1024, 256 * 1024 * 1024, DatasetPreparedInputFormatVerifierV1(prepared_format)
    if prepared_format == "synaptic-conversation-sft-row/v1":
        from tuner.dataset_prep.conversation_messages import verify_bytes
        from tuner.training.conversation_preparation import ConversationPreparedInputFormatVerifierV1
        return verify_bytes, 2 * 1024 * 1024, 2 * 1024 * 1024, ConversationPreparedInputFormatVerifierV1()
    if prepared_format == "synaptic-grpo-prompts/v1":
        from tuner.training.grpo_preparation import verify_grpo_publication_bytes, GRPOPreparedInputFormatVerifierV1
        return verify_grpo_publication_bytes, 4096, 65536, GRPOPreparedInputFormatVerifierV1()
    raise ModalPreparedCopyUnavailable("modal_prepared_format_unavailable")


def _verified_identity(verified, prepared_format):
    if prepared_format == "synaptic-grpo-prompts/v1":
        digest = verified["semantic_digest"]
        return PreparedTrainingInputIdentity("prepared://sha256/" + digest, digest,
            verified["content_digest"], verified["size_bytes"], prepared_format)
    semantic = verified.semantic_identity
    return PreparedTrainingInputIdentity("prepared://sha256/" + semantic.dataset_digest,
        semantic.dataset_digest, semantic.dataset_sha256, semantic.dataset_bytes, prepared_format)


@dataclass(frozen=True, slots=True)
class _CopiedPublicationConfig:
    identity: PreparedTrainingInputIdentity

    @property
    def normalizer_ref(self):
        return "synaptic.modal-copied-publication/v1"


class _CopiedPublicationNormalizer:
    def __init__(self, root, verifier):
        self._root, self._verifier = root, verifier
        self._authority = PosixPrivatePreparedRootAuthorityV1()
        self._attestation = self._authority.attest(root)

    def prepare(self, source, *, config, output_root):
        if type(source) is not LocalTrainingInputPathV1 or type(config) is not _CopiedPublicationConfig:
            raise TypeError("exact copied publication input required")
        self._authority.verify(self._attestation)
        directory = source.path.parent if source.path.name == "dataset.jsonl" else source.path
        if output_root != self._root or directory.parent != self._root or directory.resolve(strict=True) != directory:
            raise ValueError("copied publication is outside private root")
        verified = self._verifier.verify(directory)
        if (config.identity.ref != "prepared://sha256/" + verified.semantic_digest
                or verified.semantic_digest != config.identity.revision
                or verified.content_digest != config.identity.content_digest
                or verified.size_bytes != config.identity.size_bytes
                or self._verifier.format != config.identity.format):
            raise ValueError("copied publication differs from expected identity")
        self._authority.verify(self._attestation)
        return NormalizedTrainingInputV1(directory, self._verifier.format)


def modal_copied_preparation_ports(*, private_root, identity, train_rows=None, validation_rows=None):
    """Select existing immutable verifiers/sources; never regenerate copied rows."""
    if type(identity) is not PreparedTrainingInputIdentity:
        raise TypeError("exact prepared identity required")
    _, _, _, verifier = _format_ports(identity.format)
    if identity.format == "syntunia-sft-row/v2":
        from tuner.training.input_preparation import PublishedPreparedDatasetNormalizerV1, PublishedPreparedDatasetNormalizerConfigV1
        config = PublishedPreparedDatasetNormalizerConfigV1(identity.revision, train_rows, validation_rows)
        return PublishedPreparedDatasetNormalizerV1(authorized_root=private_root), config, verifier
    return _CopiedPublicationNormalizer(private_root, verifier), _CopiedPublicationConfig(identity), verifier


def inspect_mounted_prepared_dataset(source: Path, *, prepared_format="syntunia-sft-row/v2", snapshot_validator=None):
    """Read-only byte validation, without asserting private-root admission."""
    if os.name != "posix" or type(source) is not type(Path()) or not source.is_absolute():
        raise ModalPreparedCopyUnavailable("modal_source_inspection_unavailable")
    parser, manifest_bound, dataset_bound, _ = _format_ports(prepared_format)
    directory = source.parent if source.name == "dataset.jsonl" else source
    fd = -1
    try:
        info = directory.lstat()
        if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or directory.resolve(strict=True) != directory):
            raise ValueError
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError
        _inventory(fd)
        manifest = _read_member(fd, "manifest.json", manifest_bound)
        dataset = _read_member(fd, "dataset.jsonl", dataset_bound)
        verified = parser(directory, manifest, dataset)
        if snapshot_validator is not None:
            if not callable(snapshot_validator):
                raise TypeError("snapshot validator must be callable")
            snapshot_validator(deepcopy(verified), dataset)
        if (_read_member(fd, "manifest.json", manifest_bound) != manifest
                or _read_member(fd, "dataset.jsonl", dataset_bound) != dataset
                or os.fstat(fd).st_ino != opened.st_ino):
            raise ValueError
        _named_directory(directory, opened)
        _inventory(fd)
        return verified
    except Exception:
        raise ModalPreparedCopyUnavailable("modal_source_inspection_unavailable") from None
    finally:
        if fd >= 0:
            os.close(fd)


def _read_member(directory: int, name: str, maximum: int) -> bytes:
    member = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    try:
        info = os.fstat(member)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_size < 1 or info.st_size > maximum):
            raise ValueError
        with os.fdopen(os.dup(member), "rb") as handle:
            data = handle.read(maximum + 1)
        def identity(value):
            return value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_nlink
        if (len(data) != info.st_size or identity(os.fstat(member)) != identity(info)
                or identity(os.stat(name, dir_fd=directory, follow_symlinks=False)) != identity(info)):
            raise ValueError
        return data
    finally:
        os.close(member)


def stage_published_modal_dataset(
    source: Path, *, private_root: Path,
    expected_dataset_digest: str, expected_content_digest: str | None = None,
    prepared_format: str = "syntunia-sft-row/v2",
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
    parser, manifest_bound, dataset_bound, _ = _format_ports(prepared_format)
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
        _inventory(source_fd)
        manifest_data = _read_member(source_fd, "manifest.json", manifest_bound)
        copied_data = _read_member(source_fd, "dataset.jsonl", dataset_bound)
        if (_read_member(source_fd, "manifest.json", manifest_bound) != manifest_data
                or _read_member(source_fd, "dataset.jsonl", dataset_bound) != copied_data):
            raise ValueError
        inspected = _verified_identity(parser(source_dir, manifest_data, copied_data), prepared_format)
        if (inspected.revision != expected_dataset_digest
                or expected_content_digest is not None and inspected.content_digest != expected_content_digest):
            raise ValueError
        _named_directory(source_dir, opened)
        _inventory(source_fd)
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
        _, _, _, verifier = _format_ports(prepared_format)
        result = verifier.verify(copied)
        if (result.semantic_digest != expected_dataset_digest
                or (expected_content_digest is not None
                    and result.content_digest != expected_content_digest)
                or result.snapshot != copied_data):
            raise ValueError
        _named_directory(source_dir, opened)
        _inventory(source_fd)
        authority.verify(attestation)
        return copied / "dataset.jsonl"
    except Exception:
        raise ModalPreparedCopyUnavailable("modal_private_preparation_failed") from None
    finally:
        for fd in (destination_fd, root_fd, source_fd):
            if fd >= 0:
                os.close(fd)
