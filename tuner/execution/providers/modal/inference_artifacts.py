"""Read authenticated Modal artifact projections from one retained mount.

This module proves local data and filesystem consistency only.  A trusted
remote composition must authenticate the launch, native inventory, and exact
Volume mapping before constructing the reader.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import sys

from synaptic_tuner.api.v1.results import TrainingRunRef, VerifiedArtifact
from tuner.inference.retrieved_model import (
    MAX_ARTIFACT_BYTES,
    MAX_SEMANTIC_ARTIFACT_BYTES,
    READ,
    ROLES,
    SMALL,
    _open_root,
    _platform,
)
from tuner.inference.serving_target import _open_relative_directory

from .contracts import ArtifactMemberV1, provider_entry_identity


_MAX_TOTAL_BYTES = 2 * MAX_ARTIFACT_BYTES + 3 * MAX_SEMANTIC_ARTIFACT_BYTES
_CHUNK_BYTES = 1_048_576


class ModalInferenceArtifactError(RuntimeError):
    """Closed failure while reading one admitted mounted artifact set."""


def _run_copy(value: object) -> TrainingRunRef:
    if type(value) is not TrainingRunRef:
        raise TypeError("exact training run required")
    return TrainingRunRef.from_dict(value.to_dict())


def _artifact_copy(value: object) -> VerifiedArtifact:
    if type(value) is not VerifiedArtifact:
        raise TypeError("exact verified artifact required")
    return VerifiedArtifact.from_dict(value.to_dict())


def _member_copy(value: object) -> ArtifactMemberV1:
    if type(value) is not ArtifactMemberV1:
        raise TypeError("exact native artifact member required")
    return ArtifactMemberV1(
        value.role,
        value.path,
        value.size,
        value.sha256,
        value.provider_entry_id,
    )


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
    )


def _close_owned(*descriptors: int | None) -> None:
    """Attempt every close without replacing an active primary failure."""
    active_error = sys.exc_info()[0] is not None
    ordinary_error: BaseException | None = None
    control_error: BaseException | None = None
    for descriptor in descriptors:
        if descriptor is None:
            continue
        try:
            os.close(descriptor)
        except (KeyboardInterrupt, SystemExit) as error:
            if control_error is None:
                control_error = error
        except BaseException as error:
            if ordinary_error is None:
                ordinary_error = error
    if active_error:
        return
    if control_error is not None:
        raise control_error
    if ordinary_error is not None:
        raise ModalInferenceArtifactError("modal_inference_artifact_invalid") from None


@dataclass(frozen=True, slots=True)
class _MountedArtifactStream:
    run: TrainingRunRef
    artifact: VerifiedArtifact
    _reader: "ModalMountedInferenceArtifactReader"

    def iter_bytes(self):
        yield from self._reader._iter(self.run, self.artifact)


class ModalMountedInferenceArtifactReader:
    """Bounded byte reader for an already-authenticated native inventory."""

    __slots__ = (
        "_root",
        "_root_fd",
        "_root_identity",
        "_run",
        "_artifacts",
        "_members",
        "_artifact_volume_id",
        "_expected_artifact_volume_id",
        "_effect_id",
        "_expected_effect_id",
        "_presented_run",
        "_presented_artifacts",
        "_presented_members",
    )

    def __init__(
        self,
        *,
        root: Path,
        root_fd: int,
        run: TrainingRunRef,
        artifact_volume_id: str,
        effect_id: str,
        artifacts: tuple[VerifiedArtifact, ...],
        members: tuple[ArtifactMemberV1, ...],
    ) -> None:
        try:
            _platform()
            if (
                not isinstance(root, Path)
                or not root.is_absolute()
                or Path(os.path.normpath(root)) != root
                or type(root_fd) is not int
                or root_fd < 0
                or type(artifact_volume_id) is not str
                or not artifact_volume_id
                or type(effect_id) is not str
                or not effect_id
                or "/" in effect_id
                or type(artifacts) is not tuple
                or type(members) is not tuple
            ):
                raise ValueError("invalid mounted inference artifact inputs")
            owned_run = _run_copy(run)
            owned_artifacts = tuple(_artifact_copy(item) for item in artifacts)
            owned_members = tuple(_member_copy(item) for item in members)
            if tuple(item.role for item in owned_artifacts) != ROLES:
                raise ValueError("artifact roles are incomplete or out of order")
            if (
                any(
                    item.size_bytes <= 0
                    or item.size_bytes
                    > (
                        MAX_SEMANTIC_ARTIFACT_BYTES
                        if item.role in SMALL
                        else MAX_ARTIFACT_BYTES
                    )
                    for item in owned_artifacts
                )
                or sum(item.size_bytes for item in owned_artifacts) > _MAX_TOTAL_BYTES
            ):
                raise ValueError("artifact inventory exceeds bounds")
            if len(owned_members) != len(ROLES):
                raise ValueError("native artifact inventory is incomplete")
            members_by_role = {item.role.value: item for item in owned_members}
            if len(members_by_role) != len(ROLES):
                raise ValueError("native artifact roles are not unique")
            prefix = f"operations/{effect_id}/output/"
            for artifact in owned_artifacts:
                member = members_by_role.get(artifact.role)
                expected_path = prefix + artifact.role
                if (
                    member is None
                    or member.path != expected_path
                    or member.size != artifact.size_bytes
                    or member.sha256 != artifact.sha256
                    or member.provider_entry_id
                    != provider_entry_identity(
                        artifact_volume_id, expected_path, artifact.size_bytes
                    )
                ):
                    raise ValueError("native artifact projection differs")
            root_info = os.fstat(root_fd)
            if not stat.S_ISDIR(root_info.st_mode):
                raise ValueError("artifact root descriptor is not a directory")
            named_root = _open_root(root)
            try:
                named_info = os.fstat(named_root)
                if (named_info.st_dev, named_info.st_ino) != (
                    root_info.st_dev,
                    root_info.st_ino,
                ):
                    raise ValueError("artifact root descriptor differs from path")
            finally:
                os.close(named_root)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise ModalInferenceArtifactError(
                "modal_inference_artifact_invalid"
            ) from None
        self._root = root
        self._root_fd = root_fd
        self._root_identity = (root_info.st_dev, root_info.st_ino)
        self._run = owned_run
        self._artifacts = owned_artifacts
        self._members = tuple(sorted(owned_members, key=lambda item: item.role.value))
        self._artifact_volume_id = artifact_volume_id
        self._expected_artifact_volume_id = artifact_volume_id
        self._effect_id = effect_id
        self._expected_effect_id = effect_id
        self._presented_run = run
        self._presented_artifacts = artifacts
        self._presented_members = members

    def _inputs_unchanged(self) -> bool:
        return (
            type(self._presented_run) is TrainingRunRef
            and self._presented_run == self._run
            and type(self._presented_artifacts) is tuple
            and len(self._presented_artifacts) == len(self._artifacts)
            and all(
                type(actual) is VerifiedArtifact and actual == expected
                for actual, expected in zip(
                    self._presented_artifacts, self._artifacts, strict=True
                )
            )
            and type(self._presented_members) is tuple
            and len(self._presented_members) == len(self._members)
            and tuple(sorted(self._presented_members, key=lambda item: item.role.value))
            == self._members
            and self._artifact_volume_id == self._expected_artifact_volume_id
            and self._effect_id == self._expected_effect_id
        )

    def _root_unchanged(self) -> bool:
        info = os.fstat(self._root_fd)
        if (
            not stat.S_ISDIR(info.st_mode)
            or (
                info.st_dev,
                info.st_ino,
            )
            != self._root_identity
        ):
            return False
        named = _open_root(self._root)
        try:
            named_info = os.fstat(named)
            return (named_info.st_dev, named_info.st_ino) == self._root_identity
        finally:
            os.close(named)

    def _inventory(
        self, output_fd: int
    ) -> dict[str, tuple[int, int, int, int, int, int]]:
        expected = {item.role for item in self._artifacts}
        names = []
        with os.scandir(output_fd) as entries:
            for entry in entries:
                names.append(entry.name)
                if len(names) > len(ROLES):
                    raise ValueError("mounted output inventory has extra entries")
        if len(names) != len(expected) or set(names) != expected:
            raise ValueError("mounted output inventory differs")
        result = {}
        members = {item.role.value: item for item in self._members}
        for name in names:
            info = os.stat(name, dir_fd=output_fd, follow_symlinks=False)
            member = members[name]
            expected_path = f"operations/{self._effect_id}/output/{name}"
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_size != member.size
                or member.path != expected_path
                or provider_entry_identity(
                    self._artifact_volume_id, expected_path, info.st_size
                )
                != member.provider_entry_id
            ):
                raise ValueError("mounted output member differs")
            result[name] = _file_identity(info)
        return result

    def read_artifact(
        self, run: TrainingRunRef, artifact: VerifiedArtifact
    ) -> _MountedArtifactStream:
        try:
            requested_run = _run_copy(run)
            requested_artifact = _artifact_copy(artifact)
            if (
                not self._inputs_unchanged()
                or requested_run != self._run
                or requested_artifact not in self._artifacts
                or run != requested_run
                or artifact != requested_artifact
            ):
                raise ValueError("artifact request differs")
            return _MountedArtifactStream(
                TrainingRunRef.from_dict(requested_run.to_dict()),
                VerifiedArtifact.from_dict(requested_artifact.to_dict()),
                self,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise ModalInferenceArtifactError(
                "modal_inference_artifact_invalid"
            ) from None

    def _iter(self, run: TrainingRunRef, artifact: VerifiedArtifact):
        output_fd = None
        leaf_fd = None
        try:
            expected_run = _run_copy(run)
            expected_artifact = _artifact_copy(artifact)
            if (
                type(run) is not TrainingRunRef
                or type(artifact) is not VerifiedArtifact
                or run != expected_run
                or artifact != expected_artifact
                or expected_run != self._run
                or expected_artifact not in self._artifacts
                or not self._inputs_unchanged()
                or not self._root_unchanged()
            ):
                raise ValueError("artifact stream input differs")
            output_fd = _open_relative_directory(
                self._root_fd, f"operations/{self._effect_id}/output"
            )
            output_info = os.fstat(output_fd)
            output_identity = (output_info.st_dev, output_info.st_ino)
            before_inventory = self._inventory(output_fd)
            leaf_fd = os.open(expected_artifact.role, READ, dir_fd=output_fd)
            opened = os.fstat(leaf_fd)
            if _file_identity(opened) != before_inventory[expected_artifact.role]:
                raise ValueError("artifact changed before open")
            digest = hashlib.sha256()
            remaining = expected_artifact.size_bytes
            while remaining:
                chunk = os.read(leaf_fd, min(_CHUNK_BYTES, remaining))
                if not chunk:
                    raise ValueError("artifact stream truncated")
                remaining -= len(chunk)
                digest.update(chunk)
                yield chunk
                if (
                    run != expected_run
                    or artifact != expected_artifact
                    or not self._inputs_unchanged()
                ):
                    raise ValueError("artifact stream inputs changed")
            if os.read(leaf_fd, 1):
                raise ValueError("artifact stream overflow")
            after = os.fstat(leaf_fd)
            if (
                _file_identity(after) != _file_identity(opened)
                or digest.hexdigest() != expected_artifact.sha256
                or not self._root_unchanged()
            ):
                raise ValueError("artifact changed during read")
            if self._inventory(output_fd) != before_inventory:
                raise ValueError("mounted output inventory changed")
            current_output = _open_relative_directory(
                self._root_fd, f"operations/{self._effect_id}/output"
            )
            try:
                current_info = os.fstat(current_output)
                if (current_info.st_dev, current_info.st_ino) != output_identity:
                    raise ValueError("mounted output directory changed")
            finally:
                _close_owned(current_output)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise ModalInferenceArtifactError(
                "modal_inference_artifact_invalid"
            ) from None
        finally:
            _close_owned(leaf_fd, output_fd)


__all__: list[str] = []
