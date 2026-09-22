"""Exclusive prepared-input staging for packaged Modal training."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import BinaryIO

from synaptic_tuner.api.v1._contract import PreparedTrainingInputIdentity
from tuner.execution.foundation_v2.canonical import canonical_bytes, digest_text, safe_ref
from tuner.runtime.releases import PackagedExecutionBindingV1
from tuner.training.contracts import RetainedTrainingInputStreamLease

from .contracts import operation_path, provider_entry_identity
from .facade import ExplicitModal154ReadFacade, ModalFacadeError
from .prepared_input import MAX_MOUNTED_PREPARED_DATASET_BYTES


MODAL_PACKAGED_PREPARED_INPUT_SCHEMA = "synaptic-modal-packaged-prepared-input/v1"


@dataclass(frozen=True, slots=True)
class ModalPackagedPreparedInputDescriptor:
    identity: PreparedTrainingInputIdentity
    stage_effect_id: str
    artifact_volume_id: str
    relative_path: str

    def __post_init__(self) -> None:
        if type(self.identity) is not PreparedTrainingInputIdentity:
            raise TypeError("exact prepared input identity required")
        safe_ref(self.stage_effect_id, "stage_effect_id")
        safe_ref(self.artifact_volume_id, "artifact_volume_id")
        expected = operation_path(
            self.stage_effect_id, "input", "prepared",
            self.identity.content_digest, "payload.bin",
        )
        if self.relative_path != expected:
            raise ValueError("packaged prepared input path is not stage scoped")
        if self.identity.size_bytes > MAX_MOUNTED_PREPARED_DATASET_BYTES:
            raise ValueError("packaged prepared input exceeds the Modal staging bound")

    @classmethod
    def create(
        cls,
        execution: PackagedExecutionBindingV1,
        *,
        stage_effect_id: str,
        artifact_volume_id: str,
    ) -> "ModalPackagedPreparedInputDescriptor":
        if type(execution) is not PackagedExecutionBindingV1:
            raise TypeError("exact packaged execution binding required")
        identity = PreparedTrainingInputIdentity.from_dict(
            execution.to_dict()["prepared_input"]
        )
        return cls(
            identity, stage_effect_id, artifact_volume_id,
            operation_path(
                stage_effect_id, "input", "prepared",
                identity.content_digest, "payload.bin",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": MODAL_PACKAGED_PREPARED_INPUT_SCHEMA,
            "transport": "modal_volume_v1",
            "identity": self.identity.to_dict(),
            "stage_effect_id": self.stage_effect_id,
            "artifact_volume_id": self.artifact_volume_id,
            "relative_path": self.relative_path,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True, slots=True)
class ModalPackagedStageMaterial:
    descriptor: ModalPackagedPreparedInputDescriptor
    execution_binding_digest: str
    lease: RetainedTrainingInputStreamLease = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.descriptor) is not ModalPackagedPreparedInputDescriptor:
            raise TypeError("exact packaged input descriptor required")
        digest_text(self.execution_binding_digest, "execution_binding_digest")
        if type(self.lease) is not RetainedTrainingInputStreamLease:
            raise TypeError("exact retained prepared-input lease required")
        if self.lease.identity != self.descriptor.identity:
            raise ValueError("prepared-input lease differs from descriptor")


@dataclass(frozen=True, slots=True)
class ModalPackagedStageReceipt:
    stage_effect_id: str
    execution_binding_digest: str
    artifact_volume_id: str
    path: str
    size_bytes: int
    content_digest: str
    provider_entry_id: str

    def __post_init__(self) -> None:
        safe_ref(self.stage_effect_id, "stage_effect_id")
        safe_ref(self.artifact_volume_id, "artifact_volume_id")
        digest_text(self.execution_binding_digest, "execution_binding_digest")
        digest_text(self.content_digest, "content_digest")
        safe_ref(self.provider_entry_id, "provider_entry_id")
        if type(self.size_bytes) is not int or not 1 <= self.size_bytes <= MAX_MOUNTED_PREPARED_DATASET_BYTES:
            raise ValueError("staged prepared-input size is invalid")
        expected_path = operation_path(
            self.stage_effect_id, "input", "prepared", self.content_digest,
            "payload.bin",
        )
        if self.path != expected_path:
            raise ValueError("staged prepared-input provider path is not effect scoped")
        expected_entry = provider_entry_identity(
            self.artifact_volume_id, expected_path, self.size_bytes,
        )
        if self.provider_entry_id != expected_entry:
            raise ValueError("staged prepared-input provider entry identity is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "synaptic-modal-packaged-stage-receipt/v1",
            "stage_effect_id": self.stage_effect_id,
            "execution_binding_digest": self.execution_binding_digest,
            "artifact_volume_id": self.artifact_volume_id,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "content_digest": self.content_digest,
            "provider_entry_id": self.provider_entry_id,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @property
    def provider_ref(self) -> str:
        return safe_ref(
            "modal-packaged-stage:" + hashlib.sha256(self.canonical_bytes).hexdigest(),
            "stage_provider_ref",
        )

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ModalPackagedStageReceipt":
        if type(value) is not dict or set(value) != {
            "schema_version", "stage_effect_id", "execution_binding_digest",
            "artifact_volume_id", "path", "size_bytes", "content_digest",
            "provider_entry_id",
        } or value.get("schema_version") != "synaptic-modal-packaged-stage-receipt/v1":
            raise ValueError("packaged stage receipt is invalid")
        return cls(
            value["stage_effect_id"], value["execution_binding_digest"],
            value["artifact_volume_id"], value["path"], value["size_bytes"],
            value["content_digest"], value["provider_entry_id"],
        )  # type: ignore[arg-type]


class _HashingReader:
    def __init__(self, stream: BinaryIO, maximum: int) -> None:
        self._stream, self._maximum = stream, maximum
        self.size = 0
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        chunk = self._stream.read(size)
        if type(chunk) is not bytes:
            raise ValueError("prepared-input stream returned non-bytes")
        self.size += len(chunk)
        if self.size > self._maximum:
            raise ValueError("prepared-input stream exceeds its declared size")
        self.digest.update(chunk)
        return chunk


def prepare_modal_packaged_stage(
    execution: PackagedExecutionBindingV1,
    lease: RetainedTrainingInputStreamLease,
    *,
    stage_effect_id: str,
    artifact_volume_id: str,
) -> ModalPackagedStageMaterial:
    descriptor = ModalPackagedPreparedInputDescriptor.create(
        execution,
        stage_effect_id=stage_effect_id,
        artifact_volume_id=artifact_volume_id,
    )
    return ModalPackagedStageMaterial(descriptor, execution.binding_digest, lease)


class ModalPackagedInputStager:
    """Consume one retained stream directly into one collision-failing object."""

    def __init__(self, facade: ExplicitModal154ReadFacade) -> None:
        if type(facade) is not ExplicitModal154ReadFacade:
            raise TypeError("exact explicit Modal facade required")
        self._facade = facade

    def stage_once(self, material: ModalPackagedStageMaterial) -> ModalPackagedStageReceipt:
        if type(material) is not ModalPackagedStageMaterial:
            raise TypeError("exact packaged stage material required")
        descriptor = material.descriptor
        root = descriptor.relative_path.rsplit("/", 1)[0]
        if self._facade.list_prefix(
            descriptor.artifact_volume_id, root, max_entries=2,
        ):
            raise ModalFacadeError("modal_packaged_stage_collision")
        stream = material.lease.take_stream()
        reader = _HashingReader(stream, descriptor.identity.size_bytes)
        try:
            volume = self._facade._volume(descriptor.artifact_volume_id)
            with volume.batch_upload(force=False) as batch:
                batch.put_file(reader, descriptor.relative_path)
            if reader.read(1) != b"" or (
                reader.size,
                reader.digest.hexdigest(),
            ) != (
                descriptor.identity.size_bytes,
                descriptor.identity.content_digest,
            ):
                raise ValueError("prepared-input upload did not consume exact bytes")
        except ModalFacadeError:
            raise
        except Exception:
            raise ModalFacadeError("modal_packaged_stage_write_failed") from None
        finally:
            material.lease.close()
        digest, size = hashlib.sha256(), 0
        for chunk in self._facade.iter_complete(
            descriptor.artifact_volume_id,
            descriptor.relative_path,
            max_bytes=descriptor.identity.size_bytes,
        ):
            size += len(chunk)
            digest.update(chunk)
        listing = self._facade.list_prefix(
            descriptor.artifact_volume_id, root, max_entries=2,
        )
        provider_id = provider_entry_identity(
            descriptor.artifact_volume_id,
            descriptor.relative_path,
            descriptor.identity.size_bytes,
        )
        if (
            (size, digest.hexdigest())
            != (descriptor.identity.size_bytes, descriptor.identity.content_digest)
            or listing
            != ((descriptor.relative_path, descriptor.identity.size_bytes, provider_id),)
        ):
            raise ModalFacadeError("modal_packaged_stage_readback_mismatch")
        return ModalPackagedStageReceipt(
            descriptor.stage_effect_id,
            material.execution_binding_digest,
            descriptor.artifact_volume_id,
            descriptor.relative_path,
            descriptor.identity.size_bytes,
            descriptor.identity.content_digest,
            provider_id,
        )


__all__ = [
    "MODAL_PACKAGED_PREPARED_INPUT_SCHEMA",
    "ModalPackagedInputStager",
    "ModalPackagedPreparedInputDescriptor",
    "ModalPackagedStageMaterial",
    "ModalPackagedStageReceipt",
    "prepare_modal_packaged_stage",
]
