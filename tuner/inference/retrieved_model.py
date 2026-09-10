"""Verified SFT artifact materialization into a caller-owned private root."""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import os
import re
import secrets
import stat
import tarfile
from typing import BinaryIO
from pathlib import Path, PurePosixPath
from synaptic_tuner.api.v1.results import (
    TrainingRunRef,
    TrainingRunState,
    VerifiedArtifact,
)
from synaptic_tuner.api.v1.runs_facade import RunArtifactRequest, RunsAPI
from tuner.runtime.verification import (
    MAX_ARTIFACT_BYTES,
    MAX_SEMANTIC_ARTIFACT_BYTES,
    MAX_ARCHIVE_MEMBER_BYTES,
    MAX_ARCHIVE_EXPANDED_BYTES,
    MAX_ARCHIVE_MEMBERS,
    _validate_sft_archive_stream,
)

ROLES = (
    "final_model",
    "tokenizer",
    "training_lineage",
    "training_metrics",
    "workload_record",
)
SMALL = frozenset(ROLES[2:])
NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
DIRECTORY = getattr(os, "O_DIRECTORY", 0)
READ = (
    os.O_RDONLY | NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
)


def _platform() -> None:
    operations = (os.open, os.mkdir, os.stat, os.unlink, os.rmdir)
    if (
        not NOFOLLOW
        or not DIRECTORY
        or any(operation not in os.supports_dir_fd for operation in operations)
    ):
        raise RuntimeError(
            "secure directory-relative filesystem operations unavailable"
        )


def _open_root(path: Path) -> int:
    if not path.is_absolute() or Path(os.path.normpath(path)) != path:
        raise ValueError("private root must be absolute and lexically canonical")
    absolute = path
    current = os.open(absolute.anchor, READ | DIRECTORY)
    try:
        for part in absolute.parts[1:]:
            following = os.open(part, READ | DIRECTORY, dir_fd=current)
            os.close(current)
            current = following
        return current
    except BaseException:
        os.close(current)
        raise


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
    )


def _digest(fd: int, expected: int) -> tuple[int, str]:
    before = os.fstat(fd)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size != expected
    ):
        raise ValueError("file identity invalid")
    digest = hashlib.sha256()
    total = 0
    while True:
        chunk = os.read(fd, min(1_048_576, expected - total + 1))
        if not chunk:
            if _identity(os.fstat(fd)) != _identity(before):
                raise ValueError("file changed while reading")
            return total, digest.hexdigest()
        total += len(chunk)
        digest.update(chunk)
        if total > expected:
            raise ValueError("file grew while reading")


def _number(value: bytes) -> int:
    if value and value[0] & 128:
        raise ValueError("base-256 tar number prohibited")
    value = value.rstrip(b"\0 ")
    if any(character not in b"01234567" for character in value):
        raise ValueError("invalid tar number")
    return int(value or b"0", 8)


def _scan(stream: BinaryIO) -> tuple[tuple[str, int], ...]:
    stream.seek(0)
    names: list[str] = []
    sizes: list[int] = []
    expanded_bytes = 0
    zero_blocks = 0
    while True:
        header = stream.read(512)
        if len(header) != 512:
            raise ValueError("truncated tar")
        if header == bytes(512):
            zero_blocks += 1
            if zero_blocks == 2:
                trailer = stream.read(10_241)
                if len(trailer) > 10_240 or any(trailer):
                    raise ValueError("noncanonical tar trailer")
                break
            continue
        zero_blocks = 0
        expected_checksum = sum(header[:148]) + 256 + sum(header[156:])
        if _number(header[148:156]) != expected_checksum:
            raise ValueError("tar checksum")
        if header[156:157] not in (b"\0", b"0") or any(header[345:500]):
            raise ValueError("tar extension/link/sparse prohibited")
        name_field = header[:100]
        raw_name, separator, suffix = name_field.partition(b"\0")
        if separator and any(suffix):
            raise ValueError("tar member name has nonzero suffix bytes")
        name = raw_name.decode("utf-8", "strict")
        path = PurePosixPath(name)
        if (
            not name
            or "\\" in name
            or path.is_absolute()
            or len(path.parts) != 1
            or path.as_posix() != name
            or name in names
        ):
            raise ValueError("noncanonical tar name")
        size = _number(header[124:136])
        if not 0 < size <= MAX_ARCHIVE_MEMBER_BYTES:
            raise ValueError("tar member bound")
        names.append(name)
        sizes.append(size)
        expanded_bytes += size
        if (
            len(names) > MAX_ARCHIVE_MEMBERS
            or expanded_bytes > MAX_ARCHIVE_EXPANDED_BYTES
        ):
            raise ValueError("tar aggregate bound")
        remaining = size
        while remaining:
            chunk = stream.read(min(1_048_576, remaining))
            if not chunk:
                raise ValueError("truncated tar payload")
            remaining -= len(chunk)
        padding = (-size) % 512
        if padding:
            raw_padding = stream.read(padding)
            if len(raw_padding) != padding or any(raw_padding):
                raise ValueError("invalid tar padding")
    if not names:
        raise ValueError("empty tar")
    stream.seek(0)
    return tuple(zip(names, sizes))


@dataclass(frozen=True, slots=True)
class MaterializedFile:
    name: str
    size_bytes: int
    sha256: str
    device: int
    inode: int


@dataclass(frozen=True, slots=True, init=False)
class RetrievedSFTModel:
    run: TrainingRunRef
    artifacts: tuple[VerifiedArtifact, ...]
    root: Path
    root_identity: tuple[int, int]
    attempt: str
    identity: tuple[int, int]
    model_files: tuple[MaterializedFile, ...]
    tokenizer_files: tuple[MaterializedFile, ...]
    model_ref: str
    model_revision: str
    tokenizer_revision: str
    model_kind: str

    def __new__(cls, *args, **kwargs):
        raise TypeError("RetrievedSFTModel values are factory-issued")

    @classmethod
    def _issue(
        cls,
        run,
        artifacts,
        root,
        root_identity,
        attempt,
        identity,
        model_files,
        tokenizer_files,
        model_ref,
        model_revision,
        tokenizer_revision,
        model_kind,
    ):
        value = object.__new__(cls)
        fields = (
            ("run", run),
            ("artifacts", artifacts),
            ("root", root),
            ("root_identity", root_identity),
            ("attempt", attempt),
            ("identity", identity),
            ("model_files", model_files),
            ("tokenizer_files", tokenizer_files),
            ("model_ref", model_ref),
            ("model_revision", model_revision),
            ("tokenizer_revision", tokenizer_revision),
            ("model_kind", model_kind),
        )
        for name, item in fields:
            object.__setattr__(value, name, item)
        value.validate()
        return value

    def validate(self) -> None:
        identities = (self.root_identity, self.identity)
        if type(self.run) is not TrainingRunRef:
            raise TypeError("run must be exact TrainingRunRef")
        if type(self.artifacts) is not tuple or any(
            type(item) is not VerifiedArtifact for item in self.artifacts
        ):
            raise TypeError("artifacts must be exact VerifiedArtifact values")
        if not isinstance(self.root, Path) or type(self.attempt) is not str:
            raise TypeError("root and attempt have invalid types")
        if any(
            type(value) is not tuple
            or len(value) != 2
            or any(type(part) is not int for part in value)
            for value in identities
        ):
            raise TypeError("filesystem identities must be exact integer pairs")
        if (
            type(self.model_files) is not tuple
            or type(self.tokenizer_files) is not tuple
            or any(
                type(item) is not MaterializedFile
                for item in self.model_files + self.tokenizer_files
            )
        ):
            raise TypeError("materialized files have invalid types")
        if any(
            type(value) is not str
            for value in (
                self.model_ref,
                self.model_revision,
                self.tokenizer_revision,
                self.model_kind,
            )
        ):
            raise TypeError("model metadata must be exact strings")
        _platform()
        root_fd = _open_root(self.root)
        try:
            root_info = os.fstat(root_fd)
            if (root_info.st_dev, root_info.st_ino) != self.root_identity:
                raise ValueError("private root substituted")
            attempt_fd = os.open(self.attempt, READ | DIRECTORY, dir_fd=root_fd)
            try:
                attempt_info = os.fstat(attempt_fd)
                if (attempt_info.st_dev, attempt_info.st_ino) != self.identity:
                    raise ValueError("attempt substituted")
                expected = tuple(
                    sorted(
                        (
                            *[x.role + ".tar" for x in self.artifacts],
                            "model",
                            "tokenizer",
                        )
                    )
                )
                if tuple(sorted(os.listdir(attempt_fd))) != expected:
                    raise ValueError("attempt inventory changed")
                for artifact in self.artifacts:
                    fd = os.open(artifact.role + ".tar", READ, dir_fd=attempt_fd)
                    try:
                        info = os.fstat(fd)
                        if (
                            not stat.S_ISREG(info.st_mode)
                            or info.st_nlink != 1
                            or _digest(fd, artifact.size_bytes)
                            != (artifact.size_bytes, artifact.sha256)
                        ):
                            raise ValueError("source artifact changed")
                    finally:
                        os.close(fd)
                if _workload_model(attempt_fd) != (
                    self.model_ref,
                    self.model_revision,
                    self.tokenizer_revision,
                ):
                    raise ValueError("model target changed")
                expected_kind = (
                    "lora"
                    if any(x.name == "adapter_config.json" for x in self.model_files)
                    else "full"
                )
                if self.model_kind != expected_kind:
                    raise ValueError("model kind changed")
                for directory_name, items in (
                    ("model", self.model_files),
                    ("tokenizer", self.tokenizer_files),
                ):
                    directory_fd = os.open(
                        directory_name, READ | DIRECTORY, dir_fd=attempt_fd
                    )
                    try:
                        if tuple(sorted(os.listdir(directory_fd))) != tuple(
                            x.name for x in items
                        ):
                            raise ValueError("inventory changed")
                        for item in items:
                            fd = os.open(item.name, READ, dir_fd=directory_fd)
                            try:
                                info = os.fstat(fd)
                                if (
                                    not stat.S_ISREG(info.st_mode)
                                    or info.st_nlink != 1
                                    or (info.st_dev, info.st_ino)
                                    != (item.device, item.inode)
                                    or _digest(fd, item.size_bytes)
                                    != (item.size_bytes, item.sha256)
                                ):
                                    raise ValueError("file changed")
                            finally:
                                os.close(fd)
                    finally:
                        os.close(directory_fd)
            finally:
                os.close(attempt_fd)
        finally:
            os.close(root_fd)

    @property
    def model_path(self) -> Path:
        self.validate()
        return self.root / self.attempt / "model"

    @property
    def tokenizer_path(self) -> Path:
        self.validate()
        return self.root / self.attempt / "tokenizer"


def _stream(
    read_artifact,
    run: TrainingRunRef,
    artifact: VerifiedArtifact,
    attempt: int,
    owned_files: dict[tuple[str, ...], tuple[int, int]],
) -> None:
    expected_run = TrainingRunRef.from_dict(run.to_dict())
    expected_artifact = VerifiedArtifact.from_dict(artifact.to_dict())
    presented_run = TrainingRunRef.from_dict(run.to_dict())
    presented_artifact = VerifiedArtifact.from_dict(artifact.to_dict())
    stream = read_artifact(presented_run, presented_artifact)

    def correspondence_is_exact() -> bool:
        return (
            type(presented_run) is TrainingRunRef
            and presented_run == expected_run
            and type(presented_artifact) is VerifiedArtifact
            and presented_artifact == expected_artifact
            and type(stream.run) is TrainingRunRef
            and stream.run == expected_run
            and type(stream.artifact) is VerifiedArtifact
            and stream.artifact == expected_artifact
        )

    if not correspondence_is_exact() or not callable(stream.iter_bytes):
        raise ValueError("stream substituted")
    fd = os.open(
        artifact.role + ".tar",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | NOFOLLOW,
        0o600,
        dir_fd=attempt,
    )
    digest = hashlib.sha256()
    total = 0
    try:
        created = os.fstat(fd)
        owned_files[(artifact.role + ".tar",)] = (created.st_dev, created.st_ino)
        for chunk in stream.iter_bytes():
            if not correspondence_is_exact():
                raise ValueError("stream correspondence changed")
            if type(chunk) is not bytes or not chunk or len(chunk) > 1_048_576:
                raise ValueError("chunk invalid")
            total += len(chunk)
            if total > artifact.size_bytes:
                raise ValueError("stream overflow")
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(fd, view)
                if type(written) is not int or not 0 < written <= len(view):
                    raise OSError("short artifact write")
                view = view[written:]
        if not correspondence_is_exact():
            raise ValueError("stream correspondence changed")
        os.fsync(fd)
    finally:
        os.close(fd)
    if (total, digest.hexdigest()) != (artifact.size_bytes, artifact.sha256):
        raise ValueError("stream integrity")


def _extract(
    attempt: int,
    role: str,
    output: int,
    kind: str,
    locked_model_ref: str | None,
    owned_files: dict[tuple[str, ...], tuple[int, int]],
) -> tuple[MaterializedFile, ...]:
    fd = os.open(role + ".tar", READ, dir_fd=attempt)
    with os.fdopen(fd, "rb") as stream:
        initial = os.fstat(stream.fileno())
        if not stat.S_ISREG(initial.st_mode) or initial.st_nlink != 1:
            raise ValueError("archive identity invalid")
        plan = _scan(stream)
        names = tuple(name for name, _ in plan)
        if _validate_sft_archive_stream(
            stream, kind, locked_model_ref=locked_model_ref
        ) != (frozenset(names), True):
            raise ValueError("archive semantics")
        stream.seek(0)
        result: list[MaterializedFile] = []
        with tarfile.open(fileobj=stream, mode="r:") as archive:
            members = archive.getmembers()
            if len(members) != len(plan):
                raise ValueError("archive plan changed")
            for member, expected in zip(members, plan):
                if not member.isfile() or (member.name, member.size) != expected:
                    raise ValueError("archive plan changed")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("archive member is not a regular file")
                destination = os.open(
                    member.name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | NOFOLLOW,
                    0o600,
                    dir_fd=output,
                )
                digest = hashlib.sha256()
                total = 0
                try:
                    created = os.fstat(destination)
                    owned_files[(kind, member.name)] = (created.st_dev, created.st_ino)
                    while total < member.size:
                        chunk = source.read(min(1_048_576, member.size - total))
                        if not chunk:
                            break
                        view = memoryview(chunk)
                        while view:
                            written = os.write(destination, view)
                            if type(written) is not int or not 0 < written <= len(view):
                                raise OSError("short member write")
                            view = view[written:]
                        digest.update(chunk)
                        total += len(chunk)
                    os.fsync(destination)
                finally:
                    os.close(destination)
                if total != member.size:
                    raise ValueError("member truncated")
                info = os.stat(member.name, dir_fd=output, follow_symlinks=False)
                result.append(
                    MaterializedFile(
                        member.name,
                        total,
                        digest.hexdigest(),
                        info.st_dev,
                        info.st_ino,
                    )
                )
        if _identity(os.fstat(stream.fileno())) != _identity(initial):
            raise ValueError("archive changed during extraction")
    return tuple(sorted(result, key=lambda item: item.name))


def _workload_model(attempt: int) -> tuple[str, str, str]:
    fd = os.open("workload_record.tar", READ, dir_fd=attempt)
    try:
        initial = os.fstat(fd)
        size = initial.st_size
        if (
            not stat.S_ISREG(initial.st_mode)
            or initial.st_nlink != 1
            or not 0 < size <= MAX_SEMANTIC_ARTIFACT_BYTES
        ):
            raise ValueError("workload bound")
        raw = b""
        while len(raw) < size:
            part = os.read(fd, min(65_536, size - len(raw)))
            if not part:
                break
            raw += part
        if len(raw) != size or _identity(os.fstat(fd)) != _identity(initial):
            raise ValueError("workload changed while reading")
    finally:
        os.close(fd)
    doc = json.loads(raw.decode("utf-8"))
    if (
        json.dumps(
            doc,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        != raw
    ):
        raise ValueError("workload record is not canonical")
    model = doc["configuration"]["document"]["model"]
    if doc.get("identities", {}).get("model") != model:
        raise ValueError("workload model identity mismatch")
    values = (model["ref"], model["revision"], model["tokenizer_revision"])
    if (
        type(values[0]) is not str
        or not values[0]
        or any(
            type(value) is not str
            or re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", value) is None
            for value in values[1:]
        )
    ):
        raise ValueError("workload model target invalid")
    if values[1] != values[2]:
        raise ValueError("model and tokenizer revisions must match")
    return values


def _cleanup(
    root: int,
    name: str,
    attempt: int,
    attempt_identity: tuple[int, int],
    directories: dict[str, int],
    owned_files: dict[tuple[str, ...], tuple[int, int]],
) -> None:
    try:
        named = os.stat(name, dir_fd=root, follow_symlinks=False)
    except OSError:
        return
    if (named.st_dev, named.st_ino) != attempt_identity:
        return
    expected = {parts[0] for parts in owned_files} | set(directories)
    if set(os.listdir(attempt)) != expected:
        return
    # Prove the complete inventory before deleting anything. Directory ownership
    # alone does not authorize deleting an injected or replaced leaf.
    for directory_name, directory in directories.items():
        held = os.fstat(directory)
        named = os.stat(directory_name, dir_fd=attempt, follow_symlinks=False)
        if (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino):
            return
        expected_children = {
            parts[1]
            for parts in owned_files
            if len(parts) == 2 and parts[0] == directory_name
        }
        if set(os.listdir(directory)) != expected_children:
            return
    for parts, identity in owned_files.items():
        parent = attempt if len(parts) == 1 else directories[parts[0]]
        info = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != identity:
            return
    for parts, identity in owned_files.items():
        parent = attempt if len(parts) == 1 else directories[parts[0]]
        info = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        if (info.st_dev, info.st_ino) != identity:
            return
        os.unlink(parts[-1], dir_fd=parent)
    for directory_name in directories:
        os.rmdir(directory_name, dir_fd=attempt)
    named = os.stat(name, dir_fd=root, follow_symlinks=False)
    if (named.st_dev, named.st_ino) != attempt_identity:
        return
    os.rmdir(name, dir_fd=root)


def _materialize_admitted_sft_model(
    *,
    run: TrainingRunRef,
    artifacts: tuple[VerifiedArtifact, ...],
    root: Path,
    root_fd: int,
    read_artifact,
    maximum_artifact_bytes: int = MAX_ARTIFACT_BYTES,
    maximum_total_bytes: int = 2 * MAX_ARTIFACT_BYTES + 3 * MAX_SEMANTIC_ARTIFACT_BYTES,
) -> RetrievedSFTModel:
    """Materialize bytes admitted by an immediately composing authority.

    ``read_artifact`` is a transport capability, not authentication.  The
    public Runs adapter and the Modal worker must authenticate their evidence
    before calling this private helper.
    """
    if (
        type(run) is not TrainingRunRef
        or type(artifacts) is not tuple
        or any(type(item) is not VerifiedArtifact for item in artifacts)
        or not isinstance(root, Path)
        or type(root_fd) is not int
        or root_fd < 0
        or not callable(read_artifact)
    ):
        raise TypeError("exact inputs required")
    owned_run = TrainingRunRef.from_dict(run.to_dict())
    owned_artifacts = tuple(
        VerifiedArtifact.from_dict(artifact.to_dict()) for artifact in artifacts
    )
    if owned_run != run or owned_artifacts != artifacts:
        raise ValueError("artifact inputs changed during reconstruction")
    if (
        type(maximum_artifact_bytes) is not int
        or not 1 <= maximum_artifact_bytes <= MAX_ARTIFACT_BYTES
    ):
        raise ValueError("invalid bound")
    if (
        type(maximum_total_bytes) is not int
        or not 1
        <= maximum_total_bytes
        <= 2 * MAX_ARTIFACT_BYTES + 3 * MAX_SEMANTIC_ARTIFACT_BYTES
    ):
        raise ValueError("invalid total bound")
    _platform()
    if not root.is_absolute() or Path(os.path.normpath(root)) != root:
        raise ValueError("private root must be absolute and lexically canonical")
    if tuple(item.role for item in owned_artifacts) != ROLES:
        raise ValueError("artifacts are not the exact verified SFT result")
    if any(
        item.size_bytes <= 0
        or item.size_bytes
        > (
            MAX_SEMANTIC_ARTIFACT_BYTES
            if item.role in SMALL
            else maximum_artifact_bytes
        )
        for item in owned_artifacts
    ):
        raise ValueError("artifact bound")
    if sum(item.size_bytes for item in owned_artifacts) > maximum_total_bytes:
        raise ValueError("total bound")
    root_info = os.fstat(root_fd)
    if not stat.S_ISDIR(root_info.st_mode):
        raise ValueError("private root descriptor is not a directory")
    root_identity = (root_info.st_dev, root_info.st_ino)
    named_root_fd = _open_root(root)
    try:
        named_root = os.fstat(named_root_fd)
        if (named_root.st_dev, named_root.st_ino) != root_identity:
            raise ValueError("private root descriptor differs from path")
    finally:
        os.close(named_root_fd)
    name = ".retrieved-" + secrets.token_hex(16)
    attempt: int | None = None
    attempt_identity: tuple[int, int] | None = None
    directories: dict[str, int] = {}
    owned_files: dict[tuple[str, ...], tuple[int, int]] = {}
    try:
        os.mkdir(name, 0o700, dir_fd=root_fd)
        attempt = os.open(name, READ | DIRECTORY, dir_fd=root_fd)
        attempt_info = os.fstat(attempt)
        attempt_identity = (attempt_info.st_dev, attempt_info.st_ino)
        for artifact in owned_artifacts:
            _stream(read_artifact, owned_run, artifact, attempt, owned_files)
            if run != owned_run or artifacts != owned_artifacts:
                raise ValueError("artifact inputs changed during materialization")
        for directory_name in ("model", "tokenizer"):
            os.mkdir(directory_name, 0o700, dir_fd=attempt)
            directories[directory_name] = os.open(
                directory_name,
                READ | DIRECTORY,
                dir_fd=attempt,
            )
        target = _workload_model(attempt)
        model_files = _extract(
            attempt,
            "final_model",
            directories["model"],
            "model",
            target[0],
            owned_files,
        )
        tokenizer_files = _extract(
            attempt,
            "tokenizer",
            directories["tokenizer"],
            "tokenizer",
            None,
            owned_files,
        )
        for directory in directories.values():
            os.fsync(directory)
        if {item.name for item in model_files} & {
            item.name for item in tokenizer_files
        }:
            raise ValueError("member overlap")
        os.fsync(attempt)
        os.fsync(root_fd)
        model_kind = (
            "lora"
            if any(item.name == "adapter_config.json" for item in model_files)
            else "full"
        )
        return RetrievedSFTModel._issue(
            owned_run,
            owned_artifacts,
            root,
            root_identity,
            name,
            attempt_identity,
            model_files,
            tokenizer_files,
            *target,
            model_kind,
        )
    except BaseException:
        if attempt is not None and attempt_identity is not None:
            try:
                _cleanup(
                    root_fd,
                    name,
                    attempt,
                    attempt_identity,
                    directories,
                    owned_files,
                )
            except BaseException:
                # Preserve the original failure. Uncertain ownership leaves a
                # closed orphan rather than deleting anything unproven.
                pass
        raise
    finally:
        for directory in directories.values():
            os.close(directory)
        if attempt is not None:
            os.close(attempt)


def materialize_verified_sft_model(
    runs: RunsAPI,
    run: TrainingRunRef,
    root: Path,
    *,
    maximum_artifact_bytes: int = MAX_ARTIFACT_BYTES,
    maximum_total_bytes: int = 2 * MAX_ARTIFACT_BYTES + 3 * MAX_SEMANTIC_ARTIFACT_BYTES,
) -> RetrievedSFTModel:
    if (
        type(runs) is not RunsAPI
        or type(run) is not TrainingRunRef
        or not isinstance(root, Path)
    ):
        raise TypeError("exact inputs required")
    if (
        type(maximum_artifact_bytes) is not int
        or not 1 <= maximum_artifact_bytes <= MAX_ARTIFACT_BYTES
    ):
        raise ValueError("invalid bound")
    if (
        type(maximum_total_bytes) is not int
        or not 1
        <= maximum_total_bytes
        <= 2 * MAX_ARTIFACT_BYTES + 3 * MAX_SEMANTIC_ARTIFACT_BYTES
    ):
        raise ValueError("invalid total bound")
    _platform()
    if not root.is_absolute() or Path(os.path.normpath(root)) != root:
        raise ValueError("private root must be absolute and lexically canonical")
    root_fd = _open_root(root)
    try:
        run_snapshot = run.to_dict()
        owned_run = TrainingRunRef.from_dict(run_snapshot)

        def check_run() -> None:
            if run.to_dict() != run_snapshot or owned_run.to_dict() != run_snapshot:
                raise ValueError("run changed during admission")

        verification = runs.reverify(owned_run)
        check_run()
        if verification.run != owned_run or verification.verified is not True:
            raise ValueError("run reverification failed")
        outcome = runs.outcome(owned_run)
        check_run()
        if (
            outcome.run != owned_run
            or outcome.state is not TrainingRunState.SUCCEEDED
            or tuple(item.role for item in outcome.artifacts) != ROLES
        ):
            raise ValueError("run is not exact verified SFT result")
        artifacts = tuple(
            VerifiedArtifact.from_dict(item.to_dict()) for item in outcome.artifacts
        )

        def read_artifact(selected_run, artifact):
            check_run()
            stream = runs.artifacts(
                RunArtifactRequest(selected_run, artifact.role, artifact.size_bytes)
            )
            check_run()
            return stream

        result = _materialize_admitted_sft_model(
            run=run,
            artifacts=artifacts,
            root=root,
            root_fd=root_fd,
            read_artifact=read_artifact,
            maximum_artifact_bytes=maximum_artifact_bytes,
            maximum_total_bytes=maximum_total_bytes,
        )
        check_run()
        return result
    finally:
        os.close(root_fd)
