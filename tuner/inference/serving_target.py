"""Validated local paths needed to load one retrieved SFT model."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Protocol, runtime_checkable

from tuner.inference.retrieved_model import (
    DIRECTORY,
    READ,
    RetrievedSFTModel,
    _digest,
    _open_root,
    _platform,
)


MAX_PINNED_FILES = 20_000
MAX_PINNED_BYTES = 1 << 40
MAX_PINNED_DIRECTORIES = 20_000
MAX_PINNED_DEPTH = 64


@dataclass(frozen=True, slots=True)
class PinnedModelFile:
    relative_path: str
    size_bytes: int
    sha256: str
    device: int
    inode: int

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        if (
            type(self.relative_path) is not str
            or not self.relative_path
            or path.is_absolute()
            or path.as_posix() != self.relative_path
            or any(part in ("", ".", "..") for part in path.parts)
        ):
            raise ValueError("pinned model file path is invalid")
        if (
            type(self.size_bytes) is not int
            or not 0 <= self.size_bytes <= MAX_PINNED_BYTES
        ):
            raise ValueError("pinned model file size is invalid")
        if (
            type(self.sha256) is not str
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("pinned model file digest is invalid")
        if type(self.device) is not int or type(self.inode) is not int:
            raise TypeError("pinned model file identity is invalid")


def _identity(value: object, name: str) -> tuple[int, int]:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(part) is not int for part in value)
    ):
        raise TypeError(f"{name} must be an exact integer pair")
    return value


def _snapshot_name(value: str) -> str:
    if type(value) is not str:
        raise TypeError("snapshot must be an exact string")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ValueError("snapshot path is invalid")
    return value


def _open_relative_directory(root: int, relative: str) -> int:
    current = os.dup(root)
    try:
        for part in PurePosixPath(relative).parts:
            following = os.open(part, READ | DIRECTORY, dir_fd=current)
            os.close(current)
            current = following
        return current
    except BaseException:
        os.close(current)
        raise


def _walk_plan(
    directory: int,
    prefix: tuple[str, ...] = (),
    counters: list[int] | None = None,
) -> tuple[tuple[str, int, int, int], ...]:
    if len(prefix) > MAX_PINNED_DEPTH:
        raise ValueError("pinned model directory depth exceeds bound")
    if counters is None:
        counters = [0, 0, 0, 0]
    values: list[tuple[str, int, int, int]] = []
    with os.scandir(directory) as entries:
        names = []
        for entry in entries:
            names.append(entry.name)
            counters[0] += 1
            if counters[0] > MAX_PINNED_FILES + MAX_PINNED_DIRECTORIES:
                raise ValueError("pinned model entry count exceeds bound")
    for name in sorted(names):
        info = os.stat(name, dir_fd=directory, follow_symlinks=False)
        parts = prefix + (name,)
        if stat.S_ISDIR(info.st_mode):
            counters[1] += 1
            if counters[1] > MAX_PINNED_DIRECTORIES:
                raise ValueError("pinned model directory count exceeds bound")
            child = os.open(name, READ | DIRECTORY, dir_fd=directory)
            try:
                values.extend(_walk_plan(child, parts, counters))
            finally:
                os.close(child)
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("pinned model contains a redirected or special member")
        counters[2] += 1
        if counters[2] > MAX_PINNED_FILES:
            raise ValueError("pinned model file count exceeds bound")
        if info.st_size > MAX_PINNED_BYTES:
            raise ValueError("pinned model member exceeds size bound")
        counters[3] += info.st_size
        if counters[3] > MAX_PINNED_BYTES:
            raise ValueError("pinned model aggregate size exceeds bound")
        values.append(("/".join(parts), info.st_size, info.st_dev, info.st_ino))
    return tuple(sorted(values, key=lambda item: item[0]))


def _walk(directory: int) -> tuple[PinnedModelFile, ...]:
    plan = _walk_plan(directory)
    values = []
    for relative_path, expected_size, device, inode in plan:
        path = PurePosixPath(relative_path)
        parent_name = "/".join(path.parts[:-1])
        parent = (
            os.dup(directory)
            if not parent_name
            else _open_relative_directory(directory, parent_name)
        )
        try:
            fd = os.open(path.name, READ, dir_fd=parent)
            try:
                info = os.fstat(fd)
                if (info.st_dev, info.st_ino) != (device, inode):
                    raise ValueError("pinned model member changed")
                size, digest = _digest(fd, expected_size)
            finally:
                os.close(fd)
        finally:
            os.close(parent)
        values.append(
            PinnedModelFile(
                relative_path,
                size,
                digest,
                device,
                inode,
            )
        )
    return tuple(values)


@dataclass(frozen=True, slots=True)
class VerifiedPinnedModelSnapshot:
    model_ref: str
    revision: str
    root: Path
    snapshot: str
    root_identity: tuple[int, int]
    snapshot_identity: tuple[int, int]
    files: tuple[PinnedModelFile, ...]

    def validate(self) -> None:
        _platform()
        if type(self.model_ref) is not str or not self.model_ref:
            raise ValueError("model_ref is invalid")
        if (
            type(self.revision) is not str
            or len(self.revision) not in (40, 64)
            or any(character not in "0123456789abcdef" for character in self.revision)
        ):
            raise ValueError("revision is not pinned")
        if not isinstance(self.root, Path):
            raise TypeError("root must be a Path")
        _snapshot_name(self.snapshot)
        root_identity = _identity(self.root_identity, "root_identity")
        snapshot_identity = _identity(self.snapshot_identity, "snapshot_identity")
        if (
            type(self.files) is not tuple
            or not self.files
            or any(type(item) is not PinnedModelFile for item in self.files)
        ):
            raise TypeError("files must be exact PinnedModelFile values")
        names = tuple(item.relative_path for item in self.files)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("pinned model files must be unique and sorted")
        if (
            len(self.files) > MAX_PINNED_FILES
            or sum(item.size_bytes for item in self.files) > MAX_PINNED_BYTES
        ):
            raise ValueError("pinned model inventory exceeds bounds")
        root = _open_root(self.root)
        try:
            root_info = os.fstat(root)
            if (root_info.st_dev, root_info.st_ino) != root_identity:
                raise ValueError("pinned model root changed")
            snapshot = _open_relative_directory(root, self.snapshot)
            try:
                info = os.fstat(snapshot)
                if (info.st_dev, info.st_ino) != snapshot_identity:
                    raise ValueError("pinned model snapshot changed")
                if _walk(snapshot) != self.files:
                    raise ValueError("pinned model inventory changed")
            finally:
                os.close(snapshot)
        finally:
            os.close(root)

    @property
    def path(self) -> Path:
        self.validate()
        return self.root / self.snapshot


@runtime_checkable
class PinnedModelPreparer(Protocol):
    def prepare(
        self, *, model_ref: str, revision: str
    ) -> VerifiedPinnedModelSnapshot: ...


def capture_pinned_model_snapshot(
    *,
    model_ref: str,
    revision: str,
    root: Path,
    snapshot: str,
) -> VerifiedPinnedModelSnapshot:
    _platform()
    if type(model_ref) is not str or not model_ref:
        raise ValueError("model_ref is invalid")
    if (
        type(revision) is not str
        or len(revision) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("revision is not pinned")
    if not isinstance(root, Path):
        raise TypeError("root must be a Path")
    _snapshot_name(snapshot)
    root_fd = _open_root(root)
    try:
        root_info = os.fstat(root_fd)
        snapshot_fd = _open_relative_directory(root_fd, snapshot)
        try:
            snapshot_info = os.fstat(snapshot_fd)
            files = _walk(snapshot_fd)
        finally:
            os.close(snapshot_fd)
    finally:
        os.close(root_fd)
    value = VerifiedPinnedModelSnapshot(
        model_ref=model_ref,
        revision=revision,
        root=root,
        snapshot=snapshot,
        root_identity=(root_info.st_dev, root_info.st_ino),
        snapshot_identity=(snapshot_info.st_dev, snapshot_info.st_ino),
        files=files,
    )
    value.validate()
    return value


@dataclass(frozen=True, slots=True)
class ServingTarget:
    retrieved: RetrievedSFTModel
    base_snapshot: VerifiedPinnedModelSnapshot | None

    def validate(self) -> None:
        if type(self.retrieved) is not RetrievedSFTModel:
            raise TypeError("retrieved must be exact RetrievedSFTModel")
        self.retrieved.validate()
        if self.retrieved.model_kind == "full":
            if self.base_snapshot is not None:
                raise ValueError("full model must not have a base snapshot")
            return
        if (
            self.retrieved.model_kind != "lora"
            or type(self.base_snapshot) is not VerifiedPinnedModelSnapshot
        ):
            raise ValueError("LoRA target requires exact pinned base snapshot")
        self.base_snapshot.validate()
        if (self.base_snapshot.model_ref, self.base_snapshot.revision) != (
            self.retrieved.model_ref,
            self.retrieved.model_revision,
        ):
            raise ValueError("base snapshot identity differs from retrieved model")

    @property
    def model_path(self) -> Path:
        self.validate()
        return self.retrieved.model_path

    @property
    def tokenizer_path(self) -> Path:
        self.validate()
        return self.retrieved.tokenizer_path

    @property
    def base_model_path(self) -> Path | None:
        self.validate()
        return None if self.base_snapshot is None else self.base_snapshot.path


def prepare_serving_target(
    retrieved: RetrievedSFTModel,
    preparer: PinnedModelPreparer | None = None,
) -> ServingTarget:
    if type(retrieved) is not RetrievedSFTModel:
        raise TypeError("retrieved must be exact RetrievedSFTModel")
    retrieved.validate()
    if retrieved.model_kind == "full":
        if preparer is not None:
            raise ValueError("full model does not accept a pinned model preparer")
        target = ServingTarget(retrieved, None)
    else:
        if not isinstance(preparer, PinnedModelPreparer):
            raise TypeError("LoRA model requires PinnedModelPreparer")
        snapshot = preparer.prepare(
            model_ref=retrieved.model_ref,
            revision=retrieved.model_revision,
        )
        if type(snapshot) is not VerifiedPinnedModelSnapshot:
            raise TypeError("preparer returned invalid snapshot type")
        target = ServingTarget(retrieved, snapshot)
    target.validate()
    return target


__all__ = [
    "PinnedModelFile",
    "PinnedModelPreparer",
    "ServingTarget",
    "VerifiedPinnedModelSnapshot",
    "capture_pinned_model_snapshot",
    "prepare_serving_target",
]
