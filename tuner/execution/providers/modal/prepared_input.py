"""Opaque prepared-input binding and private worker materialization.

The payload is authorized by the retained Modal preparation and transported as
bytes.  It is never parsed, scanned for credentials, or written beneath either
source checkout.  The fixed worker materializes it only beneath the run-private
writable state root, using the existing descriptor-relative I/O primitives.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from tuner.execution.foundation_v2.canonical import canonical_bytes, parse_canonical_object
from tuner.training.contracts import PreparedTrainingInputIdentity

from .contracts import canonical_path, operation_path
from .mounted_io import claim_directory, hash_regular, write_exclusive


PRIVATE_DATASET_MEMBER = "private-dataset.jsonl"
MAX_PRIVATE_DATASET_BYTES = 2 * 1024 * 1024
MAX_MOUNTED_PREPARED_DATASET_BYTES = 64 * 1024 * 1024
PREPARED_INPUT_DESCRIPTOR_MEMBER = "prepared-input.json"
PREPARED_INPUT_DESCRIPTOR_SCHEMA = "synaptic-modal-prepared-input/v1"
PREPARED_DATASET_FORMATS = frozenset(
    {"syntunia-sft-row/v1", "syntunia-sft-row/v2"}
)
_PREPARED_REF = re.compile(r"prepared://sha256/([0-9a-f]{64})")


@dataclass(frozen=True, slots=True)
class PreparedDatasetBinding:
    prepared_digest: str
    content_digest: str
    size_bytes: int
    dataset_format: str = "syntunia-sft-row/v1"

    @property
    def relative_path(self) -> Path:
        return Path("prepared-inputs") / self.content_digest / PRIVATE_DATASET_MEMBER

    @property
    def identity(self) -> PreparedTrainingInputIdentity:
        return PreparedTrainingInputIdentity(
            ref=f"prepared://sha256/{self.prepared_digest}",
            revision=self.prepared_digest,
            content_digest=self.content_digest,
            size_bytes=self.size_bytes,
            format=self.dataset_format,
        )


@dataclass(frozen=True, slots=True)
class MountedPreparedInputDescriptor:
    identity: PreparedTrainingInputIdentity
    relative_path: str

    def __post_init__(self) -> None:
        if type(self.identity) is not PreparedTrainingInputIdentity:
            raise TypeError("exact prepared training input identity required")
        if not self.identity.size_bytes > MAX_PRIVATE_DATASET_BYTES:
            raise ValueError("mounted prepared input must exceed the inline bound")
        expected_suffix = f"/input/prepared/{self.identity.content_digest}/payload.bin"
        if (
            type(self.relative_path) is not str
            or canonical_path(self.relative_path) != self.relative_path
            or not self.relative_path.startswith("operations/")
            or not self.relative_path.endswith(expected_suffix)
            or len(self.relative_path.split("/")) != 6
        ):
            raise ValueError("mounted prepared input path is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": PREPARED_INPUT_DESCRIPTOR_SCHEMA,
            "transport": "modal_artifact_volume",
            "identity": self.identity.to_dict(),
            "relative_path": self.relative_path,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @classmethod
    def create(
        cls, identity: PreparedTrainingInputIdentity, *, stage_effect_id: str,
    ) -> "MountedPreparedInputDescriptor":
        if type(identity) is not PreparedTrainingInputIdentity:
            raise TypeError("exact prepared training input identity required")
        return cls(
            identity,
            operation_path(
                stage_effect_id, "input", "prepared", identity.content_digest,
                "payload.bin",
            ),
        )

    @classmethod
    def parse(cls, value: bytes) -> "MountedPreparedInputDescriptor":
        document = parse_canonical_object(value, name="prepared input descriptor")
        if set(document) != {
            "schema_version", "transport", "identity", "relative_path",
        } or document.get("schema_version") != PREPARED_INPUT_DESCRIPTOR_SCHEMA \
                or document.get("transport") != "modal_artifact_volume" \
                or type(document.get("identity")) is not dict:
            raise ValueError("prepared input descriptor is invalid")
        result = cls(
            PreparedTrainingInputIdentity.from_mapping(document["identity"]),
            document["relative_path"],  # type: ignore[arg-type]
        )
        if result.canonical_bytes != value:
            raise ValueError("prepared input descriptor is not canonical")
        return result


def bind_private_dataset(
    dataset: object,
    private_dataset_bytes: bytes | None,
    mounted_descriptor: MountedPreparedInputDescriptor | None = None,
) -> PreparedDatasetBinding | None:
    """Bind optional opaque bytes to the workload's exact dataset identity."""

    if not isinstance(dataset, Mapping):
        raise ValueError("workload dataset identity is malformed")
    ref = dataset.get("ref")
    if not isinstance(ref, str):
        raise ValueError("workload dataset reference is malformed")
    if ref.startswith("project://"):
        if private_dataset_bytes is not None or mounted_descriptor is not None:
            raise ValueError("project dataset must not carry private dataset bytes")
        return None
    match = _PREPARED_REF.fullmatch(ref)
    if match is None:
        raise ValueError("workload dataset reference scheme is unsupported")
    if set(dataset) != {
        "ref", "revision", "content_digest", "size_bytes", "format",
    }:
        raise ValueError("prepared dataset identity has missing or unknown fields")
    prepared_digest = match.group(1)
    content_digest = dataset.get("content_digest")
    size_bytes = dataset.get("size_bytes")
    if (
        dataset.get("revision") != prepared_digest
        or not isinstance(content_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", content_digest) is None
        or type(size_bytes) is not int
        or not 0 < size_bytes <= MAX_MOUNTED_PREPARED_DATASET_BYTES
        or dataset.get("format") not in PREPARED_DATASET_FORMATS
    ):
        raise ValueError("prepared dataset identity does not bind its private bytes")
    binding = PreparedDatasetBinding(
        prepared_digest, content_digest, size_bytes, str(dataset["format"])
    )
    if mounted_descriptor is not None:
        if private_dataset_bytes is not None or mounted_descriptor.identity != binding.identity:
            raise ValueError("prepared dataset mounted descriptor does not match")
        return binding
    if type(private_dataset_bytes) is not bytes:
        raise ValueError("prepared dataset requires exact private bytes")
    if len(private_dataset_bytes) != size_bytes:
        raise ValueError("prepared dataset identity does not bind its private bytes")
    if size_bytes > MAX_PRIVATE_DATASET_BYTES:
        raise ValueError("private dataset exceeds its byte bound")
    if hashlib.sha256(private_dataset_bytes).hexdigest() != content_digest:
        raise ValueError("prepared dataset content digest does not match")
    return binding


def prepared_dataset_path(
    state_root: Path,
    binding: PreparedDatasetBinding,
) -> Path:
    if not isinstance(state_root, Path) or not state_root.is_absolute():
        raise ValueError("prepared dataset state root must be absolute")
    if type(binding) is not PreparedDatasetBinding:
        raise TypeError("exact prepared dataset binding required")
    return state_root / binding.relative_path


def materialize_private_dataset(
    state_root: Path,
    binding: PreparedDatasetBinding,
    content: bytes,
) -> Path:
    """Exclusively materialize and reverify one opaque prepared dataset."""

    if type(content) is not bytes:
        raise TypeError("prepared dataset materialization requires exact bytes")
    if len(content) != binding.size_bytes:
        raise ValueError("prepared dataset byte count changed before materialization")
    destination = prepared_dataset_path(state_root, binding)
    claim_directory(state_root, state_root / "prepared-inputs")
    claim_directory(state_root, destination.parent)
    write_exclusive(state_root, destination, content)
    size, digest = hash_regular(
        state_root, destination, MAX_PRIVATE_DATASET_BYTES,
    )
    if (size, digest) != (binding.size_bytes, binding.content_digest):
        raise ValueError("materialized prepared dataset does not match its binding")
    return destination


__all__: list[str] = []
