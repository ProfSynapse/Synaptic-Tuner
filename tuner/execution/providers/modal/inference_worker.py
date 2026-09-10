"""Purely composed preparation of a Modal-local verified SFT serving target.

The trusted deployment owns Volume-to-mount mapping.  This module validates
the admitted launch and local descriptor identities; it does not authenticate
Modal mounts, start a server, or grant execution authority.
"""

from __future__ import annotations

import os
from pathlib import Path
import stat
import sys

from synaptic_tuner.api.v1.results import TrainingRunRef, VerifiedArtifact
from tuner.inference.retrieved_model import (
    ROLES,
    RetrievedSFTModel,
    _materialize_admitted_sft_model,
    _open_root,
    _platform,
)
from tuner.inference.serving_target import (
    PinnedModelPreparer,
    ServingTarget,
    prepare_serving_target,
)

from .contracts import ArtifactMemberV1, ArtifactRole
from .inference_artifacts import ModalMountedInferenceArtifactReader
from .inference_preparation import _validate_preparation_snapshot
from .inference_wire import (
    ModalChatWorkerExpectation,
    admit_modal_chat_launch,
)


class ModalInferenceWorkerError(RuntimeError):
    """Closed failure while preparing a Modal-local inference target."""


def _close_owned(*descriptors: int | None) -> None:
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
        raise ModalInferenceWorkerError("modal_inference_worker_invalid") from None


def _path(value: object, name: str) -> Path:
    if (
        not isinstance(value, Path)
        or not value.is_absolute()
        or Path(os.path.normpath(value)) != value
    ):
        raise ValueError(f"{name} must be an absolute canonical Path")
    return value


def _lexically_separate(paths: tuple[Path, ...]) -> None:
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError("worker roots must be distinct and nonoverlapping")


def _open_directory(path: Path) -> tuple[int, tuple[int, int]]:
    descriptor = _open_root(path)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("worker root descriptor is not a directory")
        return descriptor, (info.st_dev, info.st_ino)
    except BaseException:
        _close_owned(descriptor)
        raise


def _roots_unchanged(
    paths: tuple[Path, ...],
    descriptors: tuple[int, ...],
    identities: tuple[tuple[int, int], ...],
) -> bool:
    for path, descriptor, identity in zip(paths, descriptors, identities, strict=True):
        held = os.fstat(descriptor)
        if not stat.S_ISDIR(held.st_mode) or (held.st_dev, held.st_ino) != identity:
            return False
        current = _open_root(path)
        try:
            info = os.fstat(current)
            if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != identity:
                return False
        finally:
            _close_owned(current)
    return True


def _artifacts(source: dict[str, object]) -> tuple[VerifiedArtifact, ...]:
    values = tuple(VerifiedArtifact.from_dict(item) for item in source["artifacts"])
    by_role = {item.role: item for item in values}
    if len(by_role) != len(ROLES) or set(by_role) != set(ROLES):
        raise ValueError("authenticated artifact inventory is incomplete")
    return tuple(by_role[role] for role in ROLES)


def _members(source: dict[str, object]) -> tuple[ArtifactMemberV1, ...]:
    return tuple(
        ArtifactMemberV1(
            ArtifactRole(item["role"]),
            item["path"],
            item["size"],
            item["sha256"],
            item["provider_entry_id"],
        )
        for item in source["members"]
    )


def prepare_modal_chat_worker(
    argument: bytes,
    *,
    expectation: ModalChatWorkerExpectation,
    verifier,
    clock,
    destination: Path,
    preparer: PinnedModelPreparer | None = None,
) -> ServingTarget:
    """Prepare one verified local target after complete launch admission."""
    try:
        admission = admit_modal_chat_launch(
            argument,
            expectation=expectation,
            verifier=verifier,
            clock=clock,
        )
        owned_expectation = ModalChatWorkerExpectation(
            **{
                name: getattr(expectation, name)
                for name in expectation.__dataclass_fields__
            }
        )
        expectation_snapshot = tuple(
            getattr(expectation, name) for name in expectation.__dataclass_fields__
        )

        # Everything below may touch the local filesystem.  Admission above is
        # intentionally the sole operation before this boundary.
        _platform()
        destination = _path(destination, "destination")
        artifact_root = _path(Path(owned_expectation.artifact_root), "artifact_root")
        control_root = _path(Path(owned_expectation.control_root), "control_root")
        cache_root = _path(Path(owned_expectation.cache_root), "cache_root")
        paths = (artifact_root, control_root, cache_root, destination)
        _lexically_separate(paths)

        snapshot_bytes = admission.preparation_snapshot
        snapshot = _validate_preparation_snapshot(snapshot_bytes)
        source = snapshot["chat_input"]["source"]
        workload = snapshot["chat_input"]["workload"]
        run = TrainingRunRef.from_dict(source["run"])
        artifacts = _artifacts(source)
        members = _members(source)
        run_snapshot = run.to_dict()
        artifact_snapshots = tuple(item.to_dict() for item in artifacts)
        model_snapshot = (
            workload["model_ref"],
            workload["model_revision"],
            workload["tokenizer_revision"],
        )

        artifact_fd = control_fd = cache_fd = destination_fd = None
        try:
            artifact_fd, artifact_identity = _open_directory(artifact_root)
            control_fd, control_identity = _open_directory(control_root)
            cache_fd, cache_identity = _open_directory(cache_root)
            destination_fd, destination_identity = _open_directory(destination)
            identities = (
                artifact_identity,
                control_identity,
                cache_identity,
                destination_identity,
            )
            if len(set(identities)) != len(identities):
                raise ValueError("worker roots alias a physical directory")

            def worker_inputs_unchanged() -> bool:
                return (
                    tuple(
                        getattr(expectation, name)
                        for name in expectation.__dataclass_fields__
                    )
                    == expectation_snapshot
                    and admission.preparation_snapshot == snapshot_bytes
                    and type(run) is TrainingRunRef
                    and run.to_dict() == run_snapshot
                    and type(artifacts) is tuple
                    and tuple(item.to_dict() for item in artifacts)
                    == artifact_snapshots
                    and _roots_unchanged(
                        paths,
                        (artifact_fd, control_fd, cache_fd, destination_fd),
                        identities,
                    )
                )

            reader = ModalMountedInferenceArtifactReader(
                root=artifact_root,
                root_fd=artifact_fd,
                run=run,
                artifact_volume_id=source["artifact_volume_id"],
                effect_id=source["effect_id"],
                artifacts=artifacts,
                members=members,
            )
            retrieved = _materialize_admitted_sft_model(
                run=run,
                artifacts=artifacts,
                root=destination,
                root_fd=destination_fd,
                read_artifact=reader.read_artifact,
            )
            if type(retrieved) is not RetrievedSFTModel:
                raise TypeError("materializer returned invalid model type")
            retrieved.validate()
            if (
                not worker_inputs_unchanged()
                or retrieved.run.to_dict() != run_snapshot
                or tuple(item.to_dict() for item in retrieved.artifacts)
                != artifact_snapshots
                or (
                    retrieved.model_ref,
                    retrieved.model_revision,
                    retrieved.tokenizer_revision,
                )
                != model_snapshot
                or owned_expectation.artifact_root != artifact_root.as_posix()
                or owned_expectation.control_root != control_root.as_posix()
                or owned_expectation.cache_root != cache_root.as_posix()
            ):
                raise ValueError("materialized model differs from authenticated launch")
            if not worker_inputs_unchanged():
                raise ValueError("worker inputs or roots changed before preparation")
            target = prepare_serving_target(
                retrieved, preparer if retrieved.model_kind == "lora" else None
            )
            if type(target) is not ServingTarget:
                raise TypeError("serving preparation returned invalid target type")
            target.validate()
            if (
                not worker_inputs_unchanged()
                or target.retrieved is not retrieved
                or target.retrieved.run.to_dict() != run_snapshot
                or tuple(item.to_dict() for item in target.retrieved.artifacts)
                != artifact_snapshots
                or (
                    target.retrieved.model_ref,
                    target.retrieved.model_revision,
                    target.retrieved.tokenizer_revision,
                )
                != model_snapshot
            ):
                raise ValueError("worker inputs or roots changed")
            return target
        finally:
            _close_owned(destination_fd, cache_fd, control_fd, artifact_fd)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ModalInferenceWorkerError("modal_inference_worker_invalid") from None


__all__: list[str] = []
