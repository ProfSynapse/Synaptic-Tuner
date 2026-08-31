"""Provider-neutral, one-shot artifact publication implementation.

This module is deliberately self-contained.  Hosts own destination resolution,
authentication, spooling, persistence, and concrete publication adapters.  The
engine owns canonical binding, mutation admission, and evidence verification.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from threading import RLock
from typing import Protocol

from synaptic_tuner.api.v1.artifacts_facade import (
    ArtifactDestination,
    DestinationPage,
    PublicationPage,
    PublicationRef,
    PublicationRequest,
    PublicationResult,
    PublicationState,
    PublicationVerification,
)
from synaptic_tuner.api.v1.results import TrainingRunRef, VerifiedArtifact
from synaptic_tuner.api.v1.runs_facade import RunArtifactRequest, RunArtifactStream
from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    digest_text,
    domain_digest,
    exact_integer,
    safe_ref,
)


MAX_PAGE = 100
MAX_ARTIFACTS = 100
MAX_ARTIFACT_BYTES = 2**63 - 1
MAX_CHUNK_BYTES = 1_048_576


class PublicationCodeV1(str, Enum):
    DESTINATION_MISSING = "destination_missing"
    DESTINATION_INVALID = "destination_invalid"
    SOURCE_INVALID = "source_invalid"
    SOURCE_UNVERIFIED = "source_unverified"
    SOURCE_CONTENT_INVALID = "source_content_invalid"
    PUBLICATION_CONFLICT = "publication_conflict"
    PUBLICATION_MISSING = "publication_missing"
    EVIDENCE_INVALID = "evidence_invalid"
    PAGE_INCOMPLETE = "page_incomplete"
    STATE_CONFLICT = "state_conflict"


class PublicationErrorV1(ValueError):
    def __init__(self, code: PublicationCodeV1) -> None:
        if type(code) is not PublicationCodeV1:
            raise TypeError("code must be exact PublicationCodeV1")
        self.code = code
        super().__init__(code.value)


def _closed(code: PublicationCodeV1) -> PublicationErrorV1:
    return PublicationErrorV1(code)


def _exact_tuple(value: object, item_type: type, name: str) -> tuple:
    if type(value) is not tuple or any(type(item) is not item_type for item in value):
        raise TypeError(f"{name} must be an exact tuple of {item_type.__name__}")
    return value


def _inventory(values: tuple[VerifiedArtifact, ...]) -> tuple[VerifiedArtifact, ...]:
    artifacts = _exact_tuple(values, VerifiedArtifact, "artifacts")
    if not artifacts or len(artifacts) > MAX_ARTIFACTS:
        raise ValueError("artifacts must contain 1 through 100 entries")
    roles = tuple(item.role for item in artifacts)
    if roles != tuple(sorted(roles)) or len(roles) != len(set(roles)):
        raise ValueError("artifact roles must be unique and ascending")
    return artifacts


def _bounded_text(value: str, name: str, maximum: int = 256) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be nonempty exact text")
    if len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{name} exceeds {maximum} UTF-8 bytes")
    return value


def _artifact_inventory_digest(values: tuple[VerifiedArtifact, ...]) -> str:
    return domain_digest(
        "synaptic-publication-source-inventory/v1",
        canonical_bytes({
            "entries": [domain_digest(
                "synaptic-publication-source-entry/v1",
                canonical_bytes(item.to_dict()),
            ) for item in values]
        }),
    )


class EvidenceAuthorityPortV1(Protocol):
    def verify(self, purpose: str, payload: bytes, tag: str, key_ref: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class AuthenticatedDestinationV1:
    schema_version: str
    destination_ref: str
    display_name: str
    configuration_digest: str
    policy_digest: str
    maximum_artifact_bytes: int
    maximum_total_bytes: int
    authority_ref: str
    key_ref: str
    tag: str

    def __post_init__(self) -> None:
        if self.schema_version != "synaptic-publication-destination/v1":
            raise ValueError("unsupported destination descriptor")
        safe_ref(self.destination_ref, "destination_ref")
        _bounded_text(self.display_name, "display_name")
        digest_text(self.configuration_digest, "configuration_digest")
        digest_text(self.policy_digest, "policy_digest")
        exact_integer(self.maximum_artifact_bytes, "maximum_artifact_bytes")
        exact_integer(self.maximum_total_bytes, "maximum_total_bytes")
        if (self.maximum_artifact_bytes > MAX_ARTIFACT_BYTES
                or self.maximum_total_bytes > MAX_ARTIFACT_BYTES
                or self.maximum_artifact_bytes > self.maximum_total_bytes):
            raise ValueError("destination artifact bounds invalid")
        safe_ref(self.authority_ref, "authority_ref")
        safe_ref(self.key_ref, "key_ref")
        digest_text(self.tag, "tag")

    @property
    def payload(self) -> bytes:
        return canonical_bytes({
            "schema_version": self.schema_version,
            "destination_ref": self.destination_ref,
            "display_name": self.display_name,
            "configuration_digest": self.configuration_digest,
            "policy_digest": self.policy_digest,
            "maximum_artifact_bytes": self.maximum_artifact_bytes,
            "maximum_total_bytes": self.maximum_total_bytes,
            "authority_ref": self.authority_ref,
            "key_ref": self.key_ref,
        })

    @property
    def identity_digest(self) -> str:
        return domain_digest("synaptic-publication-destination-identity/v1", self.payload)


@dataclass(frozen=True, slots=True)
class AuthenticatedVerifiedSourceV1:
    schema_version: str
    run: TrainingRunRef
    artifacts: tuple[VerifiedArtifact, ...]
    verification_digest: str
    authority_ref: str
    key_ref: str
    tag: str

    def __post_init__(self) -> None:
        if self.schema_version != "synaptic-publication-verified-source/v1":
            raise ValueError("unsupported verified source descriptor")
        if type(self.run) is not TrainingRunRef:
            raise TypeError("run must be exact TrainingRunRef")
        object.__setattr__(self, "artifacts", _inventory(self.artifacts))
        digest_text(self.verification_digest, "verification_digest")
        safe_ref(self.authority_ref, "authority_ref")
        safe_ref(self.key_ref, "key_ref")
        digest_text(self.tag, "tag")

    @property
    def payload(self) -> bytes:
        return canonical_bytes({
            "schema_version": self.schema_version,
            "run": self.run.to_dict(),
            "artifacts": [item.to_dict() for item in self.artifacts],
            "verification_digest": self.verification_digest,
            "authority_ref": self.authority_ref,
            "key_ref": self.key_ref,
        })

    @property
    def source_identity_digest(self) -> str:
        return domain_digest("synaptic-publication-source-identity/v1", self.payload)


class VerifiedArtifactSourcePortV1(Protocol):
    def describe(self, run: TrainingRunRef) -> AuthenticatedVerifiedSourceV1: ...
    def open(self, request: RunArtifactRequest) -> RunArtifactStream: ...


class SpoolSinkPortV1(Protocol):
    def write(self, chunk: bytes) -> None: ...
    def finish(self) -> str: ...
    def abort(self) -> None: ...


class ArtifactSpoolPortV1(Protocol):
    def open(self, publication_id: str, role: str, maximum_bytes: int) -> SpoolSinkPortV1: ...


@dataclass(frozen=True, slots=True)
class SpooledArtifactV1:
    artifact: VerifiedArtifact
    spool_ref: str

    def __post_init__(self) -> None:
        if type(self.artifact) is not VerifiedArtifact:
            raise TypeError("artifact must be exact VerifiedArtifact")
        safe_ref(self.spool_ref, "spool_ref")


@dataclass(frozen=True, slots=True)
class MaterializedSourceV1:
    source_identity_digest: str
    artifacts: tuple[SpooledArtifactV1, ...]

    def __post_init__(self) -> None:
        digest_text(self.source_identity_digest, "source_identity_digest")
        values = _exact_tuple(self.artifacts, SpooledArtifactV1, "spooled artifacts")
        roles = tuple(item.artifact.role for item in values)
        if not values or roles != tuple(sorted(roles)) or len(roles) != len(set(roles)):
            raise ValueError("spooled artifacts must be nonempty, unique, and ascending")

    @property
    def inventory(self) -> tuple[VerifiedArtifact, ...]:
        return tuple(item.artifact for item in self.artifacts)


@dataclass(frozen=True, slots=True)
class PublicationCommandV1:
    schema_version: str
    publication_id: str
    mutation_id: str
    run: TrainingRunRef
    source_identity_digest: str
    source_inventory: tuple[VerifiedArtifact, ...]
    destination_ref: str
    destination_identity_digest: str
    destination_configuration_digest: str
    destination_policy_digest: str
    maximum_artifact_bytes: int
    maximum_total_bytes: int
    destination_authority_ref: str
    destination_key_ref: str
    command_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != "synaptic-publication-command/v1":
            raise ValueError("unsupported publication command")
        if type(self.run) is not TrainingRunRef:
            raise TypeError("run must be exact TrainingRunRef")
        object.__setattr__(self, "source_inventory", _inventory(self.source_inventory))
        safe_ref(self.destination_ref, "destination_ref")
        safe_ref(self.destination_authority_ref, "destination_authority_ref")
        safe_ref(self.destination_key_ref, "destination_key_ref")
        exact_integer(self.maximum_artifact_bytes, "maximum_artifact_bytes")
        exact_integer(self.maximum_total_bytes, "maximum_total_bytes")
        if (self.maximum_artifact_bytes > MAX_ARTIFACT_BYTES
                or self.maximum_total_bytes > MAX_ARTIFACT_BYTES
                or self.maximum_artifact_bytes > self.maximum_total_bytes):
            raise ValueError("command artifact bounds invalid")
        total = sum(item.size_bytes for item in self.source_inventory)
        if (any(item.size_bytes > self.maximum_artifact_bytes
                for item in self.source_inventory)
                or total > self.maximum_total_bytes):
            raise ValueError("source inventory exceeds destination bounds")
        for name in (
            "publication_id", "mutation_id", "source_identity_digest",
            "destination_identity_digest", "command_digest",
            "destination_configuration_digest", "destination_policy_digest",
        ):
            digest_text(getattr(self, name), name)
        if self.publication_id != self.expected_publication_id:
            raise ValueError("publication_id does not bind run/source/destination")
        if self.mutation_id != self.expected_mutation_id:
            raise ValueError("mutation_id does not bind publication")
        if self.command_digest != self.expected_command_digest:
            raise ValueError("command_digest does not bind command")

    @staticmethod
    def _identity_document(run, source_identity_digest, inventory, destination_ref,
                           destination_identity_digest, destination_authority_ref,
                           destination_key_ref, destination_configuration_digest,
                           destination_policy_digest, maximum_artifact_bytes,
                           maximum_total_bytes):
        return {
            "schema_version": "synaptic-publication-identity/v1",
            "run": run.to_dict(),
            "source_identity_digest": source_identity_digest,
            "source_inventory_digest": _artifact_inventory_digest(inventory),
            "source_inventory_count": len(inventory),
            "destination_ref": destination_ref,
            "destination_identity_digest": destination_identity_digest,
            "destination_configuration_digest": destination_configuration_digest,
            "destination_policy_digest": destination_policy_digest,
            "maximum_artifact_bytes": maximum_artifact_bytes,
            "maximum_total_bytes": maximum_total_bytes,
            "destination_authority_ref": destination_authority_ref,
            "destination_key_ref": destination_key_ref,
        }

    @classmethod
    def build(cls, *, run, source_identity_digest, source_inventory,
              destination_ref, destination_identity_digest,
              destination_authority_ref, destination_key_ref,
              destination_configuration_digest, destination_policy_digest,
              maximum_artifact_bytes, maximum_total_bytes):
        inventory = _inventory(source_inventory)
        identity = cls._identity_document(
            run, source_identity_digest, inventory, destination_ref,
            destination_identity_digest, destination_authority_ref,
            destination_key_ref, destination_configuration_digest,
            destination_policy_digest, maximum_artifact_bytes,
            maximum_total_bytes,
        )
        publication_id = domain_digest(
            "synaptic-bundle-publication-identity/v1", canonical_bytes(identity)
        )
        mutation_id = domain_digest(
            "synaptic-publication-mutation/v1",
            canonical_bytes({"publication_id": publication_id}),
        )
        body = {**identity, "publication_id": publication_id, "mutation_id": mutation_id}
        command_digest = domain_digest(
            "synaptic-publication-command/v1", canonical_bytes(body)
        )
        return cls(
            "synaptic-publication-command/v1", publication_id, mutation_id, run,
            source_identity_digest, inventory, destination_ref,
            destination_identity_digest, destination_configuration_digest,
            destination_policy_digest, maximum_artifact_bytes,
            maximum_total_bytes, destination_authority_ref,
            destination_key_ref, command_digest,
        )

    @property
    def expected_publication_id(self) -> str:
        return domain_digest(
            "synaptic-bundle-publication-identity/v1",
            canonical_bytes(self._identity_document(
                self.run, self.source_identity_digest, self.source_inventory,
                self.destination_ref, self.destination_identity_digest,
                self.destination_authority_ref, self.destination_key_ref,
                self.destination_configuration_digest,
                self.destination_policy_digest,
                self.maximum_artifact_bytes, self.maximum_total_bytes,
            )),
        )

    @property
    def expected_mutation_id(self) -> str:
        return domain_digest(
            "synaptic-publication-mutation/v1",
            canonical_bytes({"publication_id": self.publication_id}),
        )

    @property
    def expected_command_digest(self) -> str:
        body = {**self._identity_document(
            self.run, self.source_identity_digest, self.source_inventory,
            self.destination_ref, self.destination_identity_digest,
            self.destination_authority_ref, self.destination_key_ref,
            self.destination_configuration_digest,
            self.destination_policy_digest,
            self.maximum_artifact_bytes, self.maximum_total_bytes,
        ), "publication_id": self.publication_id, "mutation_id": self.mutation_id}
        return domain_digest("synaptic-publication-command/v1", canonical_bytes(body))


@dataclass(frozen=True, slots=True)
class DestinationArtifactV1:
    role: str
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        safe_ref(self.role, "role")
        safe_ref(self.path, "path")
        digest_text(self.sha256, "sha256")
        exact_integer(self.size_bytes, "size_bytes")
        if self.size_bytes > MAX_ARTIFACT_BYTES:
            raise ValueError("destination artifact exceeds byte limit")

    def to_dict(self) -> dict[str, object]:
        return {"role": self.role, "path": self.path, "sha256": self.sha256,
                "size_bytes": self.size_bytes}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "DestinationArtifactV1":
        return _parse_destination_artifact(value)

    @property
    def canonical_bytes(self) -> bytes:
        return _record_json_bytes(_destination_artifact_document(self))

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> "DestinationArtifactV1":
        result = _parse_destination_artifact(_parse_record_json(raw))
        if result.canonical_bytes != raw:
            raise ValueError("destination artifact failed canonical reconstruction")
        return result


@dataclass(frozen=True, slots=True)
class DestinationInventoryV1:
    artifacts: tuple[DestinationArtifactV1, ...]

    def __post_init__(self) -> None:
        values = _exact_tuple(self.artifacts, DestinationArtifactV1, "destination artifacts")
        if not values or len(values) > MAX_ARTIFACTS:
            raise ValueError("destination inventory must contain 1 through 100 entries")
        keys = tuple((item.role, item.path) for item in values)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("destination inventory must be unique and canonical")

    def to_dict(self) -> dict[str, object]:
        return _destination_inventory_document(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "DestinationInventoryV1":
        return _parse_destination_inventory(value)

    @property
    def canonical_bytes(self) -> bytes:
        return _record_json_bytes(_destination_inventory_document(self))

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> "DestinationInventoryV1":
        result = _parse_destination_inventory(_parse_record_json(raw))
        if result.canonical_bytes != raw:
            raise ValueError("destination inventory failed canonical reconstruction")
        return result

    @property
    def inventory_digest(self) -> str:
        return domain_digest(
            "synaptic-publication-destination-inventory/v1",
            canonical_bytes({"entries": [domain_digest(
                "synaptic-publication-destination-entry/v1",
                canonical_bytes(item.to_dict()),
            ) for item in self.artifacts]}),
        )


@dataclass(frozen=True, slots=True)
class AuthenticatedDestinationInventoryV1:
    inventory: DestinationInventoryV1
    publication_id: str
    command_digest: str
    mutation_id: str
    ownership_id: str
    recorded_at: str
    authority_ref: str
    key_ref: str
    tag: str

    def __post_init__(self) -> None:
        if type(self.inventory) is not DestinationInventoryV1:
            raise TypeError("inventory must be exact DestinationInventoryV1")
        for name in ("publication_id", "command_digest", "mutation_id",
                     "ownership_id", "tag"):
            digest_text(getattr(self, name), name)
        for name in ("recorded_at", "authority_ref", "key_ref"):
            safe_ref(getattr(self, name), name)

    @property
    def payload(self) -> bytes:
        return canonical_bytes({
            "schema_version": "synaptic-authenticated-destination-inventory/v1",
            "publication_id": self.publication_id,
            "command_digest": self.command_digest,
            "mutation_id": self.mutation_id,
            "ownership_id": self.ownership_id,
            "inventory_digest": self.inventory.inventory_digest,
            "inventory_count": len(self.inventory.artifacts),
            "recorded_at": self.recorded_at,
            "authority_ref": self.authority_ref,
            "key_ref": self.key_ref,
        })

    def to_dict(self) -> dict[str, object]:
        return _authenticated_inventory_document(self)

    @classmethod
    def from_dict(
        cls, value: dict[str, object]
    ) -> "AuthenticatedDestinationInventoryV1":
        return _parse_authenticated_inventory(value)

    @property
    def canonical_bytes(self) -> bytes:
        return _record_json_bytes(_authenticated_inventory_document(self))

    @classmethod
    def from_canonical_bytes(
        cls, raw: bytes
    ) -> "AuthenticatedDestinationInventoryV1":
        result = _parse_authenticated_inventory(_parse_record_json(raw))
        if result.canonical_bytes != raw:
            raise ValueError("authenticated inventory failed canonical reconstruction")
        return result


@dataclass(frozen=True, slots=True)
class TransferOwnershipV1:
    publication_id: str
    command_digest: str
    mutation_id: str
    claim_digest: str
    ownership_id: str
    issued_revision: int
    issued_at: str

    def __post_init__(self) -> None:
        for name in ("publication_id", "command_digest", "mutation_id",
                     "claim_digest", "ownership_id"):
            digest_text(getattr(self, name), name)
        exact_integer(self.issued_revision, "issued_revision")
        safe_ref(self.issued_at, "issued_at")

    @property
    def binding_digest(self) -> str:
        return domain_digest("synaptic-transfer-ownership/v1", canonical_bytes({
            "publication_id": self.publication_id,
            "command_digest": self.command_digest,
            "mutation_id": self.mutation_id,
            "claim_digest": self.claim_digest,
            "ownership_id": self.ownership_id,
            "issued_revision": self.issued_revision,
            "issued_at": self.issued_at,
        }))


@dataclass(frozen=True, slots=True)
class LookupRecoveryPermitV1:
    publication_id: str
    command_digest: str
    mutation_id: str
    claim_digest: str
    fenced_ownership_id: str
    permit_id: str
    issued_revision: int
    issued_at: str

    def __post_init__(self) -> None:
        for name in ("publication_id", "command_digest", "mutation_id",
                     "claim_digest", "fenced_ownership_id", "permit_id"):
            digest_text(getattr(self, name), name)
        exact_integer(self.issued_revision, "issued_revision")
        safe_ref(self.issued_at, "issued_at")

    @property
    def binding_digest(self) -> str:
        return domain_digest("synaptic-lookup-recovery-permit/v1", canonical_bytes({
            "publication_id": self.publication_id,
            "command_digest": self.command_digest,
            "mutation_id": self.mutation_id,
            "claim_digest": self.claim_digest,
            "fenced_ownership_id": self.fenced_ownership_id,
            "permit_id": self.permit_id,
            "issued_revision": self.issued_revision,
            "issued_at": self.issued_at,
        }))


@dataclass(frozen=True, slots=True)
class AuthenticatedPublicationReceiptV1:
    schema_version: str
    publication_id: str
    command_digest: str
    run: TrainingRunRef
    source_identity_digest: str
    destination_ref: str
    destination_identity_digest: str
    mutation_id: str
    claim_digest: str
    ownership_id: str
    inventory: AuthenticatedDestinationInventoryV1
    recorded_at: str
    authority_ref: str
    key_ref: str
    tag: str

    def __post_init__(self) -> None:
        if self.schema_version != "synaptic-publication-receipt/v1":
            raise ValueError("unsupported publication receipt")
        if (type(self.run) is not TrainingRunRef
                or type(self.inventory) is not AuthenticatedDestinationInventoryV1):
            raise TypeError("receipt types invalid")
        for name in (
            "publication_id", "command_digest", "source_identity_digest",
            "destination_identity_digest", "mutation_id", "claim_digest",
            "ownership_id", "tag",
        ):
            digest_text(getattr(self, name), name)
        for name in ("destination_ref", "recorded_at", "authority_ref", "key_ref"):
            safe_ref(getattr(self, name), name)

    @property
    def payload(self) -> bytes:
        return canonical_bytes({
            "schema_version": self.schema_version,
            "publication_id": self.publication_id,
            "command_digest": self.command_digest,
            "run": self.run.to_dict(),
            "source_identity_digest": self.source_identity_digest,
            "destination_ref": self.destination_ref,
            "destination_identity_digest": self.destination_identity_digest,
            "mutation_id": self.mutation_id,
            "claim_digest": self.claim_digest,
            "ownership_id": self.ownership_id,
            "inventory_evidence_digest": domain_digest(
                "synaptic-destination-inventory-envelope/v1", self.inventory.payload
            ),
            "recorded_at": self.recorded_at,
            "authority_ref": self.authority_ref,
            "key_ref": self.key_ref,
        })

    def to_dict(self) -> dict[str, object]:
        return _receipt_document(self)

    @classmethod
    def from_dict(
        cls, value: dict[str, object]
    ) -> "AuthenticatedPublicationReceiptV1":
        return _parse_receipt(value)

    @property
    def canonical_bytes(self) -> bytes:
        return _record_json_bytes(_receipt_document(self))

    @classmethod
    def from_canonical_bytes(
        cls, raw: bytes
    ) -> "AuthenticatedPublicationReceiptV1":
        result = _parse_receipt(_parse_record_json(raw))
        if result.canonical_bytes != raw:
            raise ValueError("publication receipt failed canonical reconstruction")
        return result


class LookupOutcomeV1(str, Enum):
    FOUND = "found"
    DEFINITELY_ABSENT = "definitely_absent"
    INDETERMINATE = "indeterminate"
    CONFLICT = "conflict"


@dataclass(frozen=True, slots=True)
class AuthenticatedPublicationTombstoneV1:
    schema_version: str
    publication_id: str
    mutation_id: str
    command_digest: str
    claim_digest: str
    destination_ref: str
    destination_identity_digest: str
    destination_configuration_digest: str
    destination_policy_digest: str
    destination_authority_ref: str
    destination_key_ref: str
    fenced_ownership_id: str
    recovery_permit_id: str
    mutation_registry_digest: str
    checked_at: str
    evidence_digest: str
    authority_ref: str
    key_ref: str
    tag: str

    def __post_init__(self) -> None:
        if self.schema_version != "synaptic-publication-tombstone/v1":
            raise ValueError("unsupported publication tombstone")
        for name in (
            "publication_id", "mutation_id", "command_digest", "claim_digest",
            "destination_identity_digest", "destination_configuration_digest",
            "destination_policy_digest", "fenced_ownership_id",
            "recovery_permit_id", "mutation_registry_digest", "evidence_digest",
            "tag",
        ):
            digest_text(getattr(self, name), name)
        for name in (
            "destination_ref", "destination_authority_ref", "destination_key_ref",
            "checked_at", "authority_ref", "key_ref",
        ):
            safe_ref(getattr(self, name), name)

    @property
    def payload(self) -> bytes:
        return canonical_bytes({
            "schema_version": self.schema_version,
            "publication_id": self.publication_id,
            "mutation_id": self.mutation_id,
            "command_digest": self.command_digest,
            "claim_digest": self.claim_digest,
            "destination_ref": self.destination_ref,
            "destination_identity_digest": self.destination_identity_digest,
            "destination_configuration_digest": self.destination_configuration_digest,
            "destination_policy_digest": self.destination_policy_digest,
            "destination_authority_ref": self.destination_authority_ref,
            "destination_key_ref": self.destination_key_ref,
            "fenced_ownership_id": self.fenced_ownership_id,
            "recovery_permit_id": self.recovery_permit_id,
            "mutation_registry_digest": self.mutation_registry_digest,
            "checked_at": self.checked_at,
            "evidence_digest": self.evidence_digest,
            "authority_ref": self.authority_ref,
            "key_ref": self.key_ref,
        })

    def to_dict(self) -> dict[str, object]:
        return _tombstone_document(self)

    @classmethod
    def from_dict(
        cls, value: dict[str, object]
    ) -> "AuthenticatedPublicationTombstoneV1":
        return _parse_tombstone(value)

    @property
    def canonical_bytes(self) -> bytes:
        return _record_json_bytes(_tombstone_document(self))

    @classmethod
    def from_canonical_bytes(
        cls, raw: bytes
    ) -> "AuthenticatedPublicationTombstoneV1":
        result = _parse_tombstone(_parse_record_json(raw))
        if result.canonical_bytes != raw:
            raise ValueError("publication tombstone failed canonical reconstruction")
        return result


@dataclass(frozen=True, slots=True)
class AuthenticatedLookupV1:
    schema_version: str
    outcome: LookupOutcomeV1
    publication_id: str
    command_digest: str
    destination_identity_digest: str
    mutation_id: str
    ownership_id: str
    recovery_permit_id: str
    mutation_registry_digest: str
    checked_at: str
    tombstone: AuthenticatedPublicationTombstoneV1 | None
    receipt: AuthenticatedPublicationReceiptV1 | None
    authority_ref: str
    key_ref: str
    tag: str

    def __post_init__(self) -> None:
        if self.schema_version != "synaptic-publication-lookup/v1":
            raise ValueError("unsupported lookup evidence")
        if type(self.outcome) is not LookupOutcomeV1:
            raise TypeError("lookup outcome type invalid")
        for name in (
            "publication_id", "command_digest", "destination_identity_digest",
            "mutation_id", "ownership_id", "recovery_permit_id",
            "mutation_registry_digest", "tag",
        ):
            digest_text(getattr(self, name), name)
        for name in ("checked_at", "authority_ref", "key_ref"):
            safe_ref(getattr(self, name), name)
        if self.outcome is LookupOutcomeV1.FOUND:
            if (type(self.receipt) is not AuthenticatedPublicationReceiptV1
                    or self.tombstone is not None):
                raise ValueError("found requires receipt and no tombstone")
        elif self.outcome is LookupOutcomeV1.DEFINITELY_ABSENT:
            if (self.receipt is not None
                    or type(self.tombstone) is not AuthenticatedPublicationTombstoneV1):
                raise ValueError("definite absence requires only a tombstone")
        elif self.tombstone is not None:
            raise ValueError("only definite absence may contain a tombstone")

    @property
    def payload(self) -> bytes:
        return canonical_bytes({
            "schema_version": self.schema_version,
            "outcome": self.outcome.value,
            "publication_id": self.publication_id,
            "command_digest": self.command_digest,
            "destination_identity_digest": self.destination_identity_digest,
            "mutation_id": self.mutation_id,
            "ownership_id": self.ownership_id,
            "recovery_permit_id": self.recovery_permit_id,
            "mutation_registry_digest": self.mutation_registry_digest,
            "checked_at": self.checked_at,
            "tombstone_digest": None if self.tombstone is None else domain_digest(
                "synaptic-publication-tombstone-envelope/v1", self.tombstone.payload
            ),
            "receipt_digest": None if self.receipt is None else domain_digest(
                "synaptic-publication-receipt-envelope/v1", self.receipt.payload
            ),
            "authority_ref": self.authority_ref,
            "key_ref": self.key_ref,
        })


class DestinationPublicationPortV1(Protocol):
    def publish_once(self, command: PublicationCommandV1,
                     source: MaterializedSourceV1,
                     ownership: TransferOwnershipV1) -> AuthenticatedPublicationReceiptV1: ...
    def lookup(self, command: PublicationCommandV1,
               permit: LookupRecoveryPermitV1) -> AuthenticatedLookupV1: ...
    def iter_bytes(self, command: PublicationCommandV1,
                   artifact: DestinationArtifactV1,
                   maximum_bytes: int) -> Iterator[bytes]: ...


class ArtifactDestinationRegistryPortV1(Protocol):
    def resolve(self, destination_ref: str) -> tuple[AuthenticatedDestinationV1,
                                                     DestinationPublicationPortV1]: ...
    def list(self, limit: int) -> tuple[tuple[AuthenticatedDestinationV1, ...], bool]: ...


class PublicationPhaseV1(str, Enum):
    CLAIMED = "claimed"
    TRANSFERRING = "transferring"
    COMMITTED = "committed"
    VERIFIED = "verified"
    AMBIGUOUS = "ambiguous"
    ABSENT = "absent"
    CONFLICT = "conflict"
    FAILED_BEFORE_EFFECT = "failed_before_effect"


class PublicationEventKindV1(str, Enum):
    CLAIMED = "claimed"
    TRANSFER_ADMITTED = "transfer_admitted"
    COMMITTED = "committed"
    VERIFIED = "verified"
    AMBIGUOUS = "ambiguous"
    ABSENCE_CONFIRMED = "absence_confirmed"
    CONFLICT_CONFIRMED = "conflict_confirmed"
    FAILED_BEFORE_EFFECT = "failed_before_effect"


@dataclass(frozen=True, slots=True)
class PublicationEventV1:
    sequence: int
    kind: PublicationEventKindV1
    timestamp: str
    prior_record_digest: str | None
    evidence_digest: str | None
    event_digest: str

    def __post_init__(self) -> None:
        exact_integer(self.sequence, "sequence")
        if type(self.kind) is not PublicationEventKindV1:
            raise TypeError("event kind must be exact PublicationEventKindV1")
        safe_ref(self.timestamp, "timestamp")
        if self.prior_record_digest is not None:
            digest_text(self.prior_record_digest, "prior_record_digest")
        if self.evidence_digest is not None:
            digest_text(self.evidence_digest, "evidence_digest")
        digest_text(self.event_digest, "event_digest")
        expected = domain_digest("synaptic-publication-event/v1", canonical_bytes({
            "sequence": self.sequence, "kind": self.kind.value,
            "timestamp": self.timestamp, "prior_record_digest": self.prior_record_digest,
            "evidence_digest": self.evidence_digest,
        }))
        if self.event_digest != expected:
            raise ValueError("event digest does not bind event")

    @classmethod
    def build(cls, sequence, kind, timestamp, prior_record_digest,
              evidence_digest=None):
        digest = domain_digest("synaptic-publication-event/v1", canonical_bytes({
            "sequence": sequence, "kind": kind.value, "timestamp": timestamp,
            "prior_record_digest": prior_record_digest,
            "evidence_digest": evidence_digest,
        }))
        return cls(sequence, kind, timestamp, prior_record_digest,
                   evidence_digest, digest)


_TRANSITIONS = {
    PublicationPhaseV1.CLAIMED: frozenset({PublicationPhaseV1.TRANSFERRING,
                                           PublicationPhaseV1.FAILED_BEFORE_EFFECT}),
    PublicationPhaseV1.TRANSFERRING: frozenset({PublicationPhaseV1.COMMITTED,
                                                PublicationPhaseV1.AMBIGUOUS}),
    PublicationPhaseV1.COMMITTED: frozenset({PublicationPhaseV1.VERIFIED,
                                             PublicationPhaseV1.AMBIGUOUS}),
    PublicationPhaseV1.AMBIGUOUS: frozenset({PublicationPhaseV1.VERIFIED,
                                             PublicationPhaseV1.ABSENT,
                                             PublicationPhaseV1.CONFLICT}),
}
_EVENT_FOR_PHASE = {
    PublicationPhaseV1.TRANSFERRING: PublicationEventKindV1.TRANSFER_ADMITTED,
    PublicationPhaseV1.COMMITTED: PublicationEventKindV1.COMMITTED,
    PublicationPhaseV1.VERIFIED: PublicationEventKindV1.VERIFIED,
    PublicationPhaseV1.AMBIGUOUS: PublicationEventKindV1.AMBIGUOUS,
    PublicationPhaseV1.ABSENT: PublicationEventKindV1.ABSENCE_CONFIRMED,
    PublicationPhaseV1.CONFLICT: PublicationEventKindV1.CONFLICT_CONFIRMED,
    PublicationPhaseV1.FAILED_BEFORE_EFFECT: PublicationEventKindV1.FAILED_BEFORE_EFFECT,
}


@dataclass(frozen=True, slots=True)
class PublicationRecordV1:
    command: PublicationCommandV1
    claim_digest: str
    phase: PublicationPhaseV1
    revision: int
    history: tuple[PublicationEventV1, ...]
    ownership_history: tuple[TransferOwnershipV1, ...]
    recovery_permits: tuple[LookupRecoveryPermitV1, ...]
    lookup_history: tuple[AuthenticatedLookupV1, ...]
    receipt: AuthenticatedPublicationReceiptV1 | None
    tombstone: AuthenticatedPublicationTombstoneV1 | None
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.command) is not PublicationCommandV1 or type(self.phase) is not PublicationPhaseV1:
            raise TypeError("record command/phase types invalid")
        digest_text(self.claim_digest, "claim_digest")
        exact_integer(self.revision, "revision")
        events = _exact_tuple(self.history, PublicationEventV1, "history")
        ownership = _exact_tuple(self.ownership_history, TransferOwnershipV1,
                                 "ownership history")
        permits = _exact_tuple(self.recovery_permits, LookupRecoveryPermitV1,
                               "recovery permits")
        outcomes = _exact_tuple(self.lookup_history, AuthenticatedLookupV1,
                                "lookup history")
        if len(events) != self.revision + 1 or tuple(e.sequence for e in events) != tuple(range(len(events))):
            raise ValueError("history/revision invariant invalid")
        if events[0].kind is not PublicationEventKindV1.CLAIMED:
            raise ValueError("history must begin with claim")
        if self.phase is PublicationPhaseV1.CLAIMED and len(events) != 1:
            raise ValueError("claimed record history invalid")
        if self.phase is not PublicationPhaseV1.CLAIMED and events[-1].kind is not _EVENT_FOR_PHASE[self.phase]:
            raise ValueError("record phase/history mismatch")
        if self.receipt is not None and type(self.receipt) is not AuthenticatedPublicationReceiptV1:
            raise TypeError("receipt type invalid")
        if (self.tombstone is not None
                and type(self.tombstone) is not AuthenticatedPublicationTombstoneV1):
            raise TypeError("tombstone type invalid")
        if self.phase in (PublicationPhaseV1.COMMITTED, PublicationPhaseV1.VERIFIED) and self.receipt is None:
            raise ValueError("committed records require receipt")
        if self.phase is PublicationPhaseV1.ABSENT and self.tombstone is None:
            raise ValueError("absent records require tombstone")
        if self.phase is PublicationPhaseV1.TRANSFERRING and not ownership:
            raise ValueError("transferring records require ownership")
        if self.phase is PublicationPhaseV1.AMBIGUOUS and not permits:
            raise ValueError("ambiguous records require recovery permit")
        if any(item.publication_id != self.command.publication_id
               or item.command_digest != self.command.command_digest
               or item.mutation_id != self.command.mutation_id
               or item.claim_digest != self.claim_digest for item in ownership):
            raise ValueError("ownership history binding invalid")
        if any(item.publication_id != self.command.publication_id
               or item.command_digest != self.command.command_digest
               or item.mutation_id != self.command.mutation_id
               or item.claim_digest != self.claim_digest for item in permits):
            raise ValueError("recovery permit binding invalid")
        if permits and permits[-1].fenced_ownership_id not in {
                item.ownership_id for item in ownership}:
            raise ValueError("recovery permit does not fence owned transfer")
        if any(item.authority_ref != self.command.destination_authority_ref
               or item.key_ref != self.command.destination_key_ref
               for item in outcomes):
            raise ValueError("lookup signer does not bind destination")
        if (self.phase in (PublicationPhaseV1.VERIFIED,
                           PublicationPhaseV1.ABSENT,
                           PublicationPhaseV1.CONFLICT)
                and permits and not outcomes):
            raise ValueError("recovery terminal state requires lookup evidence")
        digest_text(self.record_digest, "record_digest")
        if self.record_digest != self.expected_record_digest:
            raise ValueError("record digest does not bind record")

    @property
    def expected_record_digest(self) -> str:
        return domain_digest("synaptic-publication-record/v1", canonical_bytes({
            "publication_id": self.command.publication_id,
            "command_digest": self.command.command_digest,
            "claim_digest": self.claim_digest,
            "phase": self.phase.value,
            "revision": self.revision,
            "history": [event.event_digest for event in self.history],
            "ownership": [item.binding_digest for item in self.ownership_history],
            "recovery_permits": [item.binding_digest for item in self.recovery_permits],
            "lookup_history": [domain_digest(
                "synaptic-publication-lookup-envelope/v1", item.payload
            ) for item in self.lookup_history],
            "receipt_digest": None if self.receipt is None else domain_digest(
                "synaptic-publication-receipt-envelope/v1", self.receipt.payload
            ),
            "tombstone_digest": None if self.tombstone is None else domain_digest(
                "synaptic-publication-tombstone-envelope/v1", self.tombstone.payload
            ),
        }))

    @classmethod
    def claim(cls, command: PublicationCommandV1, timestamp: str):
        claim_digest = domain_digest("synaptic-publication-claim/v1", canonical_bytes({
            "publication_id": command.publication_id,
            "command_digest": command.command_digest,
            "mutation_id": command.mutation_id,
        }))
        event = PublicationEventV1.build(0, PublicationEventKindV1.CLAIMED,
                                         timestamp, None)
        return cls._create(command, claim_digest, PublicationPhaseV1.CLAIMED,
                           (event,), (), (), (), None, None)

    @classmethod
    def _create(cls, command, claim_digest, phase, history, ownership_history,
                recovery_permits, lookup_history, receipt, tombstone):
        revision = len(history) - 1
        document = {
            "publication_id": command.publication_id,
            "command_digest": command.command_digest,
            "claim_digest": claim_digest,
            "phase": phase.value,
            "revision": revision,
            "history": [item.event_digest for item in history],
            "ownership": [item.binding_digest for item in ownership_history],
            "recovery_permits": [item.binding_digest for item in recovery_permits],
            "lookup_history": [domain_digest(
                "synaptic-publication-lookup-envelope/v1", item.payload
            ) for item in lookup_history],
            "receipt_digest": None if receipt is None else domain_digest(
                "synaptic-publication-receipt-envelope/v1", receipt.payload
            ),
            "tombstone_digest": None if tombstone is None else domain_digest(
                "synaptic-publication-tombstone-envelope/v1", tombstone.payload
            ),
        }
        return cls(command, claim_digest, phase, revision, history,
                   ownership_history, recovery_permits, lookup_history,
                   receipt, tombstone,
                   domain_digest("synaptic-publication-record/v1",
                                 canonical_bytes(document)))

    def transition(self, phase: PublicationPhaseV1, timestamp: str):
        """Construct only the non-privileged pre-effect failure descendant."""
        if self.phase is not PublicationPhaseV1.CLAIMED or phase is not PublicationPhaseV1.FAILED_BEFORE_EFFECT:
            raise _closed(PublicationCodeV1.STATE_CONFLICT)
        return self._advance(phase, timestamp)

    @property
    def canonical_bytes(self) -> bytes:
        """Return the complete, bounded canonical persistence envelope."""
        return _publication_record_bytes(self)

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> "PublicationRecordV1":
        """Reconstruct and fully revalidate a persisted publication record."""
        return _parse_publication_record(raw)

    def _advance(self, phase, timestamp, *, ownership=None, permit=None,
                 outcome=None, receipt=None, tombstone=None):
        if phase not in _TRANSITIONS.get(self.phase, frozenset()):
            raise _closed(PublicationCodeV1.STATE_CONFLICT)
        if receipt is None:
            receipt = self.receipt
        if tombstone is None:
            tombstone = self.tombstone
        ownership_history = self.ownership_history + (() if ownership is None else (ownership,))
        recovery_permits = self.recovery_permits + (() if permit is None else (permit,))
        lookup_history = self.lookup_history + (() if outcome is None else (outcome,))
        evidence_parts = []
        if ownership is not None:
            evidence_parts.append(ownership.binding_digest)
        if permit is not None:
            evidence_parts.append(permit.binding_digest)
        if outcome is not None:
            evidence_parts.append(domain_digest(
                "synaptic-publication-lookup-envelope/v1", outcome.payload))
        if receipt is not None and receipt != self.receipt:
            evidence_parts.append(domain_digest(
                "synaptic-publication-receipt-envelope/v1", receipt.payload))
        if tombstone is not None and tombstone != self.tombstone:
            evidence_parts.append(domain_digest(
                "synaptic-publication-tombstone-envelope/v1", tombstone.payload))
        evidence_digest = None if not evidence_parts else domain_digest(
            "synaptic-publication-transition-evidence/v1",
            canonical_bytes({"digests": evidence_parts}),
        )
        event = PublicationEventV1.build(
            self.revision + 1, _EVENT_FOR_PHASE[phase], timestamp,
            self.record_digest, evidence_digest,
        )
        history = self.history + (event,)
        return self._create(self.command, self.claim_digest, phase, history,
                            ownership_history, recovery_permits, lookup_history, receipt,
                            tombstone)


_MAX_PUBLICATION_RECORD_BYTES = 1_048_576


def _record_fields(value: object, expected: frozenset[str], name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact object")
    keys = tuple(dict.keys(value))
    if any(type(key) is not str for key in keys):
        raise TypeError(f"{name} field names must be exact strings")
    if frozenset(keys) != expected:
        raise ValueError(f"{name} contains missing or unknown fields")
    return {key: dict.__getitem__(value, key) for key in keys}


def _record_array(value: object, name: str) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"{name} must be an exact array")
    return value


def _record_text(document: dict[str, object], field: str, name: str) -> str:
    value = dict.__getitem__(document, field)
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    return value


def _record_integer(document: dict[str, object], field: str, name: str) -> int:
    value = dict.__getitem__(document, field)
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    return value


def _record_json_bytes(value: dict[str, object]) -> bytes:
    try:
        raw = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("publication record is not canonical JSON") from exc
    if not raw or len(raw) > _MAX_PUBLICATION_RECORD_BYTES:
        raise ValueError("publication record exceeds the persistence limit")
    return raw


def _parse_record_json(raw: bytes) -> dict[str, object]:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_PUBLICATION_RECORD_BYTES:
        raise ValueError("publication record must be bounded exact bytes")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("publication record contains duplicate keys")
            result[key] = value
        return result

    def reject_number(_: str) -> object:
        raise ValueError("publication record contains a non-integer number")

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=unique_object,
            parse_constant=reject_number, parse_float=reject_number,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("publication record must be canonical UTF-8 JSON") from exc
    if type(value) is not dict or _record_json_bytes(value) != raw:
        raise ValueError("publication record must be a canonical JSON object")
    return value


def _command_document(value: PublicationCommandV1) -> dict[str, object]:
    return {
        "schema_version": value.schema_version,
        "publication_id": value.publication_id,
        "mutation_id": value.mutation_id,
        "run": value.run.to_dict(),
        "source_identity_digest": value.source_identity_digest,
        "source_inventory": [item.to_dict() for item in value.source_inventory],
        "destination_ref": value.destination_ref,
        "destination_identity_digest": value.destination_identity_digest,
        "destination_configuration_digest": value.destination_configuration_digest,
        "destination_policy_digest": value.destination_policy_digest,
        "maximum_artifact_bytes": value.maximum_artifact_bytes,
        "maximum_total_bytes": value.maximum_total_bytes,
        "destination_authority_ref": value.destination_authority_ref,
        "destination_key_ref": value.destination_key_ref,
        "command_digest": value.command_digest,
    }


_COMMAND_FIELDS = frozenset({
    "schema_version", "publication_id", "mutation_id", "run",
    "source_identity_digest", "source_inventory", "destination_ref",
    "destination_identity_digest", "destination_configuration_digest",
    "destination_policy_digest", "maximum_artifact_bytes",
    "maximum_total_bytes", "destination_authority_ref",
    "destination_key_ref", "command_digest",
})


def _parse_command(value: object) -> PublicationCommandV1:
    doc = _record_fields(value, _COMMAND_FIELDS, "publication command")
    run = _record_fields(doc["run"], frozenset({"run_id", "project_ref"}), "run")
    inventory = tuple(
        VerifiedArtifact.from_dict(_record_fields(
            item, frozenset({"role", "sha256", "size_bytes"}),
            "source artifact",
        ))
        for item in _record_array(doc["source_inventory"], "source inventory")
    )
    return PublicationCommandV1(
        doc["schema_version"], doc["publication_id"], doc["mutation_id"],
        TrainingRunRef.from_dict(run), doc["source_identity_digest"], inventory,
        doc["destination_ref"], doc["destination_identity_digest"],
        doc["destination_configuration_digest"], doc["destination_policy_digest"],
        doc["maximum_artifact_bytes"], doc["maximum_total_bytes"],
        doc["destination_authority_ref"], doc["destination_key_ref"],
        doc["command_digest"],
    )


def _event_document(value: PublicationEventV1) -> dict[str, object]:
    return {
        "sequence": value.sequence, "kind": value.kind.value,
        "timestamp": value.timestamp,
        "prior_record_digest": value.prior_record_digest,
        "evidence_digest": value.evidence_digest,
        "event_digest": value.event_digest,
    }


def _parse_event(value: object) -> PublicationEventV1:
    doc = _record_fields(value, frozenset({
        "sequence", "kind", "timestamp", "prior_record_digest",
        "evidence_digest", "event_digest",
    }), "publication event")
    return PublicationEventV1(
        doc["sequence"], PublicationEventKindV1(doc["kind"]), doc["timestamp"],
        doc["prior_record_digest"], doc["evidence_digest"], doc["event_digest"],
    )


def _ownership_document(value: TransferOwnershipV1) -> dict[str, object]:
    return {
        "publication_id": value.publication_id,
        "command_digest": value.command_digest,
        "mutation_id": value.mutation_id,
        "claim_digest": value.claim_digest,
        "ownership_id": value.ownership_id,
        "issued_revision": value.issued_revision,
        "issued_at": value.issued_at,
    }


_OWNERSHIP_FIELDS = frozenset({
    "publication_id", "command_digest", "mutation_id", "claim_digest",
    "ownership_id", "issued_revision", "issued_at",
})


def _parse_ownership(value: object) -> TransferOwnershipV1:
    doc = _record_fields(value, _OWNERSHIP_FIELDS, "transfer ownership")
    return TransferOwnershipV1(
        doc["publication_id"], doc["command_digest"], doc["mutation_id"],
        doc["claim_digest"], doc["ownership_id"], doc["issued_revision"],
        doc["issued_at"],
    )


def _permit_document(value: LookupRecoveryPermitV1) -> dict[str, object]:
    return {
        "publication_id": value.publication_id,
        "command_digest": value.command_digest,
        "mutation_id": value.mutation_id,
        "claim_digest": value.claim_digest,
        "fenced_ownership_id": value.fenced_ownership_id,
        "permit_id": value.permit_id,
        "issued_revision": value.issued_revision,
        "issued_at": value.issued_at,
    }


_PERMIT_FIELDS = frozenset({
    "publication_id", "command_digest", "mutation_id", "claim_digest",
    "fenced_ownership_id", "permit_id", "issued_revision", "issued_at",
})


def _parse_permit(value: object) -> LookupRecoveryPermitV1:
    doc = _record_fields(value, _PERMIT_FIELDS, "recovery permit")
    return LookupRecoveryPermitV1(
        doc["publication_id"], doc["command_digest"], doc["mutation_id"],
        doc["claim_digest"], doc["fenced_ownership_id"], doc["permit_id"],
        doc["issued_revision"], doc["issued_at"],
    )


def _destination_artifact_document(value: DestinationArtifactV1) -> dict[str, object]:
    return value.to_dict()


def _parse_destination_artifact(value: object) -> DestinationArtifactV1:
    doc = _record_fields(value, frozenset({
        "role", "path", "sha256", "size_bytes",
    }), "destination artifact")
    return DestinationArtifactV1(
        _record_text(doc, "role", "destination artifact role"),
        _record_text(doc, "path", "destination artifact path"),
        _record_text(doc, "sha256", "destination artifact sha256"),
        _record_integer(doc, "size_bytes", "destination artifact size_bytes"),
    )


def _destination_inventory_document(value: DestinationInventoryV1) -> dict[str, object]:
    return {"artifacts": [_destination_artifact_document(item) for item in value.artifacts]}


def _parse_destination_inventory(value: object) -> DestinationInventoryV1:
    doc = _record_fields(value, frozenset({"artifacts"}), "destination inventory")
    return DestinationInventoryV1(tuple(
        _parse_destination_artifact(item)
        for item in _record_array(doc["artifacts"], "destination artifacts")
    ))


def _authenticated_inventory_document(
    value: AuthenticatedDestinationInventoryV1,
) -> dict[str, object]:
    return {
        "inventory": _destination_inventory_document(value.inventory),
        "publication_id": value.publication_id,
        "command_digest": value.command_digest,
        "mutation_id": value.mutation_id,
        "ownership_id": value.ownership_id,
        "recorded_at": value.recorded_at,
        "authority_ref": value.authority_ref,
        "key_ref": value.key_ref,
        "tag": value.tag,
    }


_AUTHENTICATED_INVENTORY_FIELDS = frozenset({
    "inventory", "publication_id", "command_digest", "mutation_id",
    "ownership_id", "recorded_at", "authority_ref", "key_ref", "tag",
})


def _parse_authenticated_inventory(value: object) -> AuthenticatedDestinationInventoryV1:
    doc = _record_fields(
        value, _AUTHENTICATED_INVENTORY_FIELDS,
        "authenticated destination inventory",
    )
    return AuthenticatedDestinationInventoryV1(
        _parse_destination_inventory(dict.__getitem__(doc, "inventory")),
        _record_text(doc, "publication_id", "inventory publication_id"),
        _record_text(doc, "command_digest", "inventory command_digest"),
        _record_text(doc, "mutation_id", "inventory mutation_id"),
        _record_text(doc, "ownership_id", "inventory ownership_id"),
        _record_text(doc, "recorded_at", "inventory recorded_at"),
        _record_text(doc, "authority_ref", "inventory authority_ref"),
        _record_text(doc, "key_ref", "inventory key_ref"),
        _record_text(doc, "tag", "inventory tag"),
    )


def _receipt_document(value: AuthenticatedPublicationReceiptV1) -> dict[str, object]:
    return {
        "schema_version": value.schema_version,
        "publication_id": value.publication_id,
        "command_digest": value.command_digest,
        "run": value.run.to_dict(),
        "source_identity_digest": value.source_identity_digest,
        "destination_ref": value.destination_ref,
        "destination_identity_digest": value.destination_identity_digest,
        "mutation_id": value.mutation_id,
        "claim_digest": value.claim_digest,
        "ownership_id": value.ownership_id,
        "inventory": _authenticated_inventory_document(value.inventory),
        "recorded_at": value.recorded_at,
        "authority_ref": value.authority_ref,
        "key_ref": value.key_ref,
        "tag": value.tag,
    }


_RECEIPT_FIELDS = frozenset({
    "schema_version", "publication_id", "command_digest", "run",
    "source_identity_digest", "destination_ref", "destination_identity_digest",
    "mutation_id", "claim_digest", "ownership_id", "inventory",
    "recorded_at", "authority_ref", "key_ref", "tag",
})


def _parse_receipt(value: object) -> AuthenticatedPublicationReceiptV1:
    doc = _record_fields(value, _RECEIPT_FIELDS, "publication receipt")
    run = _record_fields(
        dict.__getitem__(doc, "run"),
        frozenset({"run_id", "project_ref"}),
        "receipt run",
    )
    return AuthenticatedPublicationReceiptV1(
        _record_text(doc, "schema_version", "receipt schema_version"),
        _record_text(doc, "publication_id", "receipt publication_id"),
        _record_text(doc, "command_digest", "receipt command_digest"),
        TrainingRunRef(
            _record_text(run, "run_id", "receipt run_id"),
            _record_text(run, "project_ref", "receipt project_ref"),
        ),
        _record_text(doc, "source_identity_digest", "receipt source_identity_digest"),
        _record_text(doc, "destination_ref", "receipt destination_ref"),
        _record_text(
            doc, "destination_identity_digest", "receipt destination_identity_digest",
        ),
        _record_text(doc, "mutation_id", "receipt mutation_id"),
        _record_text(doc, "claim_digest", "receipt claim_digest"),
        _record_text(doc, "ownership_id", "receipt ownership_id"),
        _parse_authenticated_inventory(dict.__getitem__(doc, "inventory")),
        _record_text(doc, "recorded_at", "receipt recorded_at"),
        _record_text(doc, "authority_ref", "receipt authority_ref"),
        _record_text(doc, "key_ref", "receipt key_ref"),
        _record_text(doc, "tag", "receipt tag"),
    )


_TOMBSTONE_FIELDS = frozenset({
    "schema_version", "publication_id", "mutation_id", "command_digest",
    "claim_digest", "destination_ref", "destination_identity_digest",
    "destination_configuration_digest", "destination_policy_digest",
    "destination_authority_ref", "destination_key_ref", "fenced_ownership_id",
    "recovery_permit_id", "mutation_registry_digest", "checked_at",
    "evidence_digest", "authority_ref", "key_ref", "tag",
})


def _tombstone_document(value: AuthenticatedPublicationTombstoneV1) -> dict[str, object]:
    return {name: getattr(value, name) for name in _TOMBSTONE_FIELDS}


def _parse_tombstone(value: object) -> AuthenticatedPublicationTombstoneV1:
    doc = _record_fields(value, _TOMBSTONE_FIELDS, "publication tombstone")
    return AuthenticatedPublicationTombstoneV1(*(
        _record_text(doc, name, f"tombstone {name}") for name in (
            "schema_version", "publication_id", "mutation_id", "command_digest",
            "claim_digest", "destination_ref", "destination_identity_digest",
            "destination_configuration_digest", "destination_policy_digest",
            "destination_authority_ref", "destination_key_ref",
            "fenced_ownership_id", "recovery_permit_id",
            "mutation_registry_digest", "checked_at", "evidence_digest",
            "authority_ref", "key_ref", "tag",
        )
    ))


_LOOKUP_FIELDS = frozenset({
    "schema_version", "outcome", "publication_id", "command_digest",
    "destination_identity_digest", "mutation_id", "ownership_id",
    "recovery_permit_id", "mutation_registry_digest", "checked_at",
    "tombstone", "receipt", "authority_ref", "key_ref", "tag",
})


def _lookup_document(value: AuthenticatedLookupV1) -> dict[str, object]:
    return {
        "schema_version": value.schema_version,
        "outcome": value.outcome.value,
        "publication_id": value.publication_id,
        "command_digest": value.command_digest,
        "destination_identity_digest": value.destination_identity_digest,
        "mutation_id": value.mutation_id,
        "ownership_id": value.ownership_id,
        "recovery_permit_id": value.recovery_permit_id,
        "mutation_registry_digest": value.mutation_registry_digest,
        "checked_at": value.checked_at,
        "tombstone": None if value.tombstone is None else _tombstone_document(value.tombstone),
        "receipt": None if value.receipt is None else _receipt_document(value.receipt),
        "authority_ref": value.authority_ref,
        "key_ref": value.key_ref,
        "tag": value.tag,
    }


def _parse_lookup(value: object) -> AuthenticatedLookupV1:
    doc = _record_fields(value, _LOOKUP_FIELDS, "authenticated lookup")
    return AuthenticatedLookupV1(
        doc["schema_version"], LookupOutcomeV1(doc["outcome"]),
        doc["publication_id"], doc["command_digest"],
        doc["destination_identity_digest"], doc["mutation_id"],
        doc["ownership_id"], doc["recovery_permit_id"],
        doc["mutation_registry_digest"], doc["checked_at"],
        None if doc["tombstone"] is None else _parse_tombstone(doc["tombstone"]),
        None if doc["receipt"] is None else _parse_receipt(doc["receipt"]),
        doc["authority_ref"], doc["key_ref"], doc["tag"],
    )


_RECORD_FIELDS = frozenset({
    "schema_version", "command", "claim_digest", "phase", "revision",
    "history", "ownership_history", "recovery_permits", "lookup_history",
    "receipt", "tombstone", "record_digest",
})


def _publication_record_document(value: PublicationRecordV1) -> dict[str, object]:
    if type(value) is not PublicationRecordV1:
        raise TypeError("record must be exact PublicationRecordV1")
    return {
        "schema_version": "synaptic-publication-record-envelope/v1",
        "command": _command_document(value.command),
        "claim_digest": value.claim_digest,
        "phase": value.phase.value,
        "revision": value.revision,
        "history": [_event_document(item) for item in value.history],
        "ownership_history": [_ownership_document(item) for item in value.ownership_history],
        "recovery_permits": [_permit_document(item) for item in value.recovery_permits],
        "lookup_history": [_lookup_document(item) for item in value.lookup_history],
        "receipt": None if value.receipt is None else _receipt_document(value.receipt),
        "tombstone": None if value.tombstone is None else _tombstone_document(value.tombstone),
        "record_digest": value.record_digest,
    }


def _publication_record_bytes(value: PublicationRecordV1) -> bytes:
    return _record_json_bytes(_publication_record_document(value))


def _parse_publication_record(raw: bytes) -> PublicationRecordV1:
    doc = _record_fields(_parse_record_json(raw), _RECORD_FIELDS, "publication record")
    if doc["schema_version"] != "synaptic-publication-record-envelope/v1":
        raise ValueError("unsupported publication record envelope")
    record = PublicationRecordV1(
        _parse_command(doc["command"]), doc["claim_digest"],
        PublicationPhaseV1(doc["phase"]), doc["revision"],
        tuple(_parse_event(item) for item in _record_array(doc["history"], "history")),
        tuple(_parse_ownership(item) for item in _record_array(
            doc["ownership_history"], "ownership history")),
        tuple(_parse_permit(item) for item in _record_array(
            doc["recovery_permits"], "recovery permits")),
        tuple(_parse_lookup(item) for item in _record_array(
            doc["lookup_history"], "lookup history")),
        None if doc["receipt"] is None else _parse_receipt(doc["receipt"]),
        None if doc["tombstone"] is None else _parse_tombstone(doc["tombstone"]),
        doc["record_digest"],
    )
    if record.canonical_bytes != raw:
        raise ValueError("publication record reconstruction mismatch")
    return record


class TransferDispositionV1(str, Enum):
    ACQUIRED = "acquired"
    ACTIVE = "active"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class TransferAdmissionV1:
    disposition: TransferDispositionV1
    record: PublicationRecordV1
    ownership: TransferOwnershipV1 | None


class RecoveryDispositionV1(str, Enum):
    ACTIVE = "active"
    PERMITTED = "permitted"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class RecoveryDecisionV1:
    disposition: RecoveryDispositionV1
    record: PublicationRecordV1
    permit: LookupRecoveryPermitV1 | None


class PublicationStorePortV1(Protocol):
    def claim(self, record: PublicationRecordV1) -> tuple[PublicationRecordV1, bool]: ...
    def get(self, publication_id: str) -> PublicationRecordV1 | None: ...
    def compare_and_swap(self, expected_record_digest: str,
                         descendant: PublicationRecordV1) -> bool: ...
    def begin_transfer(self, publication_id: str, expected_record_digest: str,
                       timestamp: str) -> TransferAdmissionV1: ...
    def complete_transfer(self, ownership: TransferOwnershipV1,
                          receipt: AuthenticatedPublicationReceiptV1,
                          verified: bool, timestamp: str) -> PublicationRecordV1: ...
    def relinquish_uncertain(self, ownership: TransferOwnershipV1,
                             timestamp: str) -> RecoveryDecisionV1: ...
    def recover_transfer(self, publication_id: str, command_digest: str,
                         timestamp: str) -> RecoveryDecisionV1: ...
    def finalize_recovery(self, permit: LookupRecoveryPermitV1,
                          phase: PublicationPhaseV1,
                          timestamp: str,
                          outcome: AuthenticatedLookupV1,
                          receipt: AuthenticatedPublicationReceiptV1 | None = None,
                          tombstone: AuthenticatedPublicationTombstoneV1 | None = None) -> PublicationRecordV1: ...
    def list(self, destination_ref: str, limit: int) -> tuple[tuple[PublicationRecordV1, ...], bool]: ...


class PublicationTransitionKernelV1:
    """Pure publication transitions shared by volatile and durable stores."""

    @staticmethod
    def claim(
        current: PublicationRecordV1 | None,
        record: PublicationRecordV1,
    ) -> tuple[PublicationRecordV1, bool]:
        if (type(record) is not PublicationRecordV1
                or record.phase is not PublicationPhaseV1.CLAIMED):
            raise TypeError("claim requires exact claimed record")
        if current is None:
            return record, True
        if type(current) is not PublicationRecordV1:
            raise TypeError("current must be exact PublicationRecordV1 or None")
        if current.command.command_digest != record.command.command_digest:
            raise _closed(PublicationCodeV1.PUBLICATION_CONFLICT)
        return current, False

    @staticmethod
    def compare_and_swap(
        current: PublicationRecordV1 | None,
        expected_record_digest: str,
        descendant: PublicationRecordV1,
    ) -> tuple[PublicationRecordV1 | None, bool]:
        digest_text(expected_record_digest, "expected_record_digest")
        if type(descendant) is not PublicationRecordV1:
            raise TypeError("descendant must be exact PublicationRecordV1")
        if current is not None and type(current) is not PublicationRecordV1:
            raise TypeError("current must be exact PublicationRecordV1 or None")
        if current is None or current.record_digest != expected_record_digest:
            return current, False
        if (
            descendant.revision != current.revision + 1
            or descendant.history[:-1] != current.history
            or descendant.history[-1].prior_record_digest != current.record_digest
            or descendant.command != current.command
            or descendant.claim_digest != current.claim_digest
            or current.phase is not PublicationPhaseV1.CLAIMED
            or descendant.phase is not PublicationPhaseV1.FAILED_BEFORE_EFFECT
            or descendant.ownership_history != current.ownership_history
            or descendant.recovery_permits != current.recovery_permits
            or descendant.lookup_history != current.lookup_history
            or descendant.receipt is not current.receipt
            or descendant.tombstone is not current.tombstone
        ):
            raise _closed(PublicationCodeV1.STATE_CONFLICT)
        return descendant, True

    @staticmethod
    def _ownership(
        current: PublicationRecordV1, timestamp: str, nonce: int,
    ) -> TransferOwnershipV1:
        exact_integer(nonce, "ownership nonce", minimum=1)
        ownership_id = domain_digest(
            "synaptic-transfer-owner-id/v1",
            canonical_bytes({
                "record_digest": current.record_digest,
                "nonce": nonce,
                "timestamp": timestamp,
            }),
        )
        return TransferOwnershipV1(
            current.command.publication_id, current.command.command_digest,
            current.command.mutation_id, current.claim_digest, ownership_id,
            current.revision + 1, timestamp,
        )

    @staticmethod
    def _permit(
        current: PublicationRecordV1,
        ownership: TransferOwnershipV1,
        timestamp: str,
    ) -> LookupRecoveryPermitV1:
        permit_id = domain_digest(
            "synaptic-lookup-permit-id/v1",
            canonical_bytes({
                "record_digest": current.record_digest,
                "fenced_ownership_id": ownership.ownership_id,
                "revision": current.revision + 1,
                "timestamp": timestamp,
            }),
        )
        return LookupRecoveryPermitV1(
            current.command.publication_id, current.command.command_digest,
            current.command.mutation_id, current.claim_digest,
            ownership.ownership_id, permit_id, current.revision + 1, timestamp,
        )

    @classmethod
    def begin_transfer(
        cls,
        current: PublicationRecordV1 | None,
        publication_id: str,
        expected_record_digest: str,
        timestamp: str,
        nonce: int,
    ) -> TransferAdmissionV1:
        digest_text(publication_id, "publication_id")
        digest_text(expected_record_digest, "expected_record_digest")
        safe_ref(timestamp, "timestamp")
        exact_integer(nonce, "ownership nonce", minimum=1)
        if current is None:
            raise _closed(PublicationCodeV1.PUBLICATION_MISSING)
        if type(current) is not PublicationRecordV1:
            raise TypeError("current must be exact PublicationRecordV1 or None")
        if current.command.publication_id != publication_id:
            raise _closed(PublicationCodeV1.PUBLICATION_MISSING)
        if current.phase is PublicationPhaseV1.CLAIMED:
            if current.record_digest != expected_record_digest:
                return TransferAdmissionV1(
                    TransferDispositionV1.ACTIVE, current, None,
                )
            ownership = cls._ownership(current, timestamp, nonce)
            descendant = current._advance(
                PublicationPhaseV1.TRANSFERRING, timestamp,
                ownership=ownership,
            )
            return TransferAdmissionV1(
                TransferDispositionV1.ACQUIRED, descendant, ownership,
            )
        if current.phase in (
            PublicationPhaseV1.TRANSFERRING, PublicationPhaseV1.COMMITTED,
        ):
            return TransferAdmissionV1(
                TransferDispositionV1.ACTIVE, current,
                current.ownership_history[-1],
            )
        return TransferAdmissionV1(
            TransferDispositionV1.TERMINAL, current, None,
        )

    @staticmethod
    def _ownership_matches(
        current: PublicationRecordV1, ownership: TransferOwnershipV1,
    ) -> bool:
        return (
            bool(current.ownership_history)
            and current.ownership_history[-1] == ownership
            and ownership.publication_id == current.command.publication_id
            and ownership.command_digest == current.command.command_digest
            and ownership.mutation_id == current.command.mutation_id
            and ownership.claim_digest == current.claim_digest
        )

    @staticmethod
    def _receipt_structurally_matches(
        current: PublicationRecordV1,
        ownership: TransferOwnershipV1,
        receipt: AuthenticatedPublicationReceiptV1,
    ) -> bool:
        inventory = receipt.inventory
        return (
            _receipt_matches(receipt, current.command, current.claim_digest,
                             ownership)
            and inventory.publication_id == current.command.publication_id
            and inventory.command_digest == current.command.command_digest
            and inventory.mutation_id == current.command.mutation_id
            and inventory.ownership_id == ownership.ownership_id
            and inventory.authority_ref
            == current.command.destination_authority_ref
            and inventory.key_ref == current.command.destination_key_ref
            and receipt.authority_ref
            == current.command.destination_authority_ref
            and receipt.key_ref == current.command.destination_key_ref
        )

    @classmethod
    def complete_transfer(
        cls,
        current: PublicationRecordV1 | None,
        ownership: TransferOwnershipV1,
        receipt: AuthenticatedPublicationReceiptV1,
        verified: bool,
        timestamp: str,
        *,
        ownership_active: bool,
    ) -> PublicationRecordV1:
        if (type(ownership) is not TransferOwnershipV1
                or type(receipt) is not AuthenticatedPublicationReceiptV1):
            raise TypeError("exact ownership and receipt required")
        if type(verified) is not bool or type(ownership_active) is not bool:
            raise TypeError("verified/ownership_active must be exact bool")
        if current is not None and type(current) is not PublicationRecordV1:
            raise TypeError("current must be exact PublicationRecordV1 or None")
        if (current is None or not ownership_active
                or not cls._ownership_matches(current, ownership)
                or not cls._receipt_structurally_matches(
                    current, ownership, receipt)):
            raise _closed(PublicationCodeV1.STATE_CONFLICT)
        if not verified and current.phase is PublicationPhaseV1.TRANSFERRING:
            return current._advance(
                PublicationPhaseV1.COMMITTED, timestamp, receipt=receipt,
            )
        if (verified and current.phase is PublicationPhaseV1.COMMITTED
                and current.receipt == receipt):
            return current._advance(
                PublicationPhaseV1.VERIFIED, timestamp, receipt=receipt,
            )
        raise _closed(PublicationCodeV1.STATE_CONFLICT)

    @classmethod
    def _fence(
        cls,
        current: PublicationRecordV1,
        ownership: TransferOwnershipV1,
        timestamp: str,
    ) -> RecoveryDecisionV1:
        permit = cls._permit(current, ownership, timestamp)
        descendant = current._advance(
            PublicationPhaseV1.AMBIGUOUS, timestamp, permit=permit,
        )
        return RecoveryDecisionV1(
            RecoveryDispositionV1.PERMITTED, descendant, permit,
        )

    @classmethod
    def relinquish_uncertain(
        cls,
        current: PublicationRecordV1 | None,
        ownership: TransferOwnershipV1,
        timestamp: str,
        *,
        ownership_active: bool,
    ) -> RecoveryDecisionV1:
        if type(ownership) is not TransferOwnershipV1:
            raise TypeError("exact ownership required")
        if type(ownership_active) is not bool:
            raise TypeError("ownership_active must be exact bool")
        if current is not None and type(current) is not PublicationRecordV1:
            raise TypeError("current must be exact PublicationRecordV1 or None")
        if (current is None or not ownership_active
                or not cls._ownership_matches(current, ownership)
                or current.phase not in (
                    PublicationPhaseV1.TRANSFERRING,
                    PublicationPhaseV1.COMMITTED,
                )):
            raise _closed(PublicationCodeV1.STATE_CONFLICT)
        return cls._fence(current, ownership, timestamp)

    @classmethod
    def recover_transfer(
        cls,
        current: PublicationRecordV1 | None,
        publication_id: str,
        command_digest: str,
        timestamp: str,
        *,
        ownership_active: bool,
    ) -> RecoveryDecisionV1:
        digest_text(publication_id, "publication_id")
        digest_text(command_digest, "command_digest")
        if type(ownership_active) is not bool:
            raise TypeError("ownership_active must be exact bool")
        if current is not None and type(current) is not PublicationRecordV1:
            raise TypeError("current must be exact PublicationRecordV1 or None")
        if (current is None
                or current.command.publication_id != publication_id
                or current.command.command_digest != command_digest):
            raise _closed(PublicationCodeV1.PUBLICATION_MISSING)
        if current.phase in (
            PublicationPhaseV1.TRANSFERRING, PublicationPhaseV1.COMMITTED,
        ):
            ownership = current.ownership_history[-1]
            if ownership_active:
                return RecoveryDecisionV1(
                    RecoveryDispositionV1.ACTIVE, current, None,
                )
            return cls._fence(current, ownership, timestamp)
        if current.phase is PublicationPhaseV1.AMBIGUOUS:
            return RecoveryDecisionV1(
                RecoveryDispositionV1.PERMITTED, current,
                current.recovery_permits[-1],
            )
        return RecoveryDecisionV1(
            RecoveryDispositionV1.TERMINAL, current, None,
        )

    @classmethod
    def finalize_recovery(
        cls,
        current: PublicationRecordV1 | None,
        permit: LookupRecoveryPermitV1,
        phase: PublicationPhaseV1,
        timestamp: str,
        outcome: AuthenticatedLookupV1,
        *,
        receipt: AuthenticatedPublicationReceiptV1 | None = None,
        tombstone: AuthenticatedPublicationTombstoneV1 | None = None,
    ) -> PublicationRecordV1:
        if type(permit) is not LookupRecoveryPermitV1:
            raise TypeError("exact recovery permit required")
        if type(outcome) is not AuthenticatedLookupV1:
            raise TypeError("exact lookup outcome required")
        if phase not in (
            PublicationPhaseV1.VERIFIED, PublicationPhaseV1.ABSENT,
            PublicationPhaseV1.CONFLICT,
        ):
            raise _closed(PublicationCodeV1.STATE_CONFLICT)
        if current is not None and type(current) is not PublicationRecordV1:
            raise TypeError("current must be exact PublicationRecordV1 or None")
        if (current is None or current.phase is not PublicationPhaseV1.AMBIGUOUS
                or not current.recovery_permits
                or current.recovery_permits[-1] != permit
                or outcome.authority_ref
                != current.command.destination_authority_ref
                or outcome.key_ref != current.command.destination_key_ref
                or outcome.publication_id != current.command.publication_id
                or outcome.command_digest != current.command.command_digest
                or outcome.destination_identity_digest
                != current.command.destination_identity_digest
                or outcome.mutation_id != current.command.mutation_id
                or outcome.ownership_id != permit.fenced_ownership_id
                or outcome.recovery_permit_id != permit.permit_id):
            raise _closed(PublicationCodeV1.STATE_CONFLICT)
        ownership = current.ownership_history[-1]
        if phase is PublicationPhaseV1.VERIFIED:
            if (type(receipt) is not AuthenticatedPublicationReceiptV1
                    or outcome.outcome is not LookupOutcomeV1.FOUND
                    or outcome.receipt != receipt
                    or not cls._receipt_structurally_matches(
                        current, ownership, receipt)
                    or (current.receipt is not None
                        and current.receipt != receipt)):
                raise _closed(PublicationCodeV1.STATE_CONFLICT)
        elif phase is PublicationPhaseV1.ABSENT:
            if (type(tombstone) is not AuthenticatedPublicationTombstoneV1
                    or outcome.outcome is not LookupOutcomeV1.DEFINITELY_ABSENT
                    or outcome.tombstone != tombstone
                    or outcome.mutation_registry_digest
                    != tombstone.mutation_registry_digest
                    or not _tombstone_matches(
                        tombstone, current.command, current.claim_digest,
                        ownership, permit)):
                raise _closed(PublicationCodeV1.STATE_CONFLICT)
        return current._advance(
            phase, timestamp, outcome=outcome, receipt=receipt,
            tombstone=tombstone,
        )


class StrongInMemoryPublicationStoreV1:
    """Thread-safe reference store enforcing exact claims and direct descendants."""

    def __init__(self) -> None:
        self._records: dict[str, PublicationRecordV1] = {}
        self._live_owners: set[str] = set()
        self._nonce = 0
        self._lock = RLock()

    def claim(self, record: PublicationRecordV1):
        if (type(record) is not PublicationRecordV1
                or record.phase is not PublicationPhaseV1.CLAIMED):
            raise TypeError("claim requires exact claimed record")
        with self._lock:
            current, created = PublicationTransitionKernelV1.claim(
                self._records.get(record.command.publication_id), record,
            )
            if created:
                self._records[record.command.publication_id] = current
            return current, created

    def get(self, publication_id: str):
        digest_text(publication_id, "publication_id")
        with self._lock:
            return self._records.get(publication_id)

    def compare_and_swap(self, expected_record_digest: str, descendant: PublicationRecordV1):
        if type(descendant) is not PublicationRecordV1:
            raise TypeError("descendant must be exact PublicationRecordV1")
        with self._lock:
            current, changed = PublicationTransitionKernelV1.compare_and_swap(
                self._records.get(descendant.command.publication_id),
                expected_record_digest, descendant,
            )
            if changed:
                self._records[descendant.command.publication_id] = current
            return changed

    def begin_transfer(self, publication_id, expected_record_digest, timestamp):
        with self._lock:
            current = self._records.get(publication_id)
            if (current is not None
                    and current.phase is PublicationPhaseV1.CLAIMED
                    and current.record_digest == expected_record_digest):
                self._nonce += 1
            admission = PublicationTransitionKernelV1.begin_transfer(
                current, publication_id, expected_record_digest, timestamp,
                max(1, self._nonce),
            )
            if admission.disposition is TransferDispositionV1.ACQUIRED:
                self._records[publication_id] = admission.record
                self._live_owners.add(admission.ownership.ownership_id)
            return admission

    def complete_transfer(self, ownership, receipt, verified, timestamp):
        if (type(ownership) is not TransferOwnershipV1
                or type(receipt) is not AuthenticatedPublicationReceiptV1):
            raise TypeError("exact ownership and receipt required")
        with self._lock:
            current = self._records.get(ownership.publication_id)
            descendant = PublicationTransitionKernelV1.complete_transfer(
                current, ownership, receipt, verified, timestamp,
                ownership_active=ownership.ownership_id in self._live_owners,
            )
            self._records[ownership.publication_id] = descendant
            if verified:
                self._live_owners.discard(ownership.ownership_id)
            return descendant

    def relinquish_uncertain(self, ownership, timestamp):
        if type(ownership) is not TransferOwnershipV1:
            raise TypeError("exact ownership required")
        with self._lock:
            current = self._records.get(ownership.publication_id)
            decision = PublicationTransitionKernelV1.relinquish_uncertain(
                current, ownership, timestamp,
                ownership_active=ownership.ownership_id in self._live_owners,
            )
            self._records[ownership.publication_id] = decision.record
            self._live_owners.discard(ownership.ownership_id)
            return decision

    def recover_transfer(self, publication_id, command_digest, timestamp):
        with self._lock:
            current = self._records.get(publication_id)
            active = (
                current is not None
                and bool(current.ownership_history)
                and current.ownership_history[-1].ownership_id
                in self._live_owners
            )
            decision = PublicationTransitionKernelV1.recover_transfer(
                current, publication_id, command_digest, timestamp,
                ownership_active=active,
            )
            if decision.record is not current:
                self._records[publication_id] = decision.record
                if current is not None and current.ownership_history:
                    self._live_owners.discard(
                        current.ownership_history[-1].ownership_id,
                    )
            return decision

    def mark_orphaned(self, ownership):
        """Reference-store crash hook; durable hosts derive this from fencing."""
        if type(ownership) is not TransferOwnershipV1:
            raise TypeError("exact ownership required")
        with self._lock:
            self._live_owners.discard(ownership.ownership_id)

    def finalize_recovery(self, permit, phase, timestamp, outcome,
                          receipt=None, tombstone=None):
        if type(permit) is not LookupRecoveryPermitV1:
            raise TypeError("exact recovery permit required")
        with self._lock:
            current = self._records.get(permit.publication_id)
            descendant = PublicationTransitionKernelV1.finalize_recovery(
                current, permit, phase, timestamp, outcome,
                receipt=receipt, tombstone=tombstone,
            )
            self._records[permit.publication_id] = descendant
            return descendant

    def list(self, destination_ref: str, limit: int):
        safe_ref(destination_ref, "destination_ref")
        if type(limit) is not int or limit != MAX_PAGE + 1:
            raise ValueError("publication list probe limit must be 101")
        with self._lock:
            values = tuple(sorted(
                (item for item in self._records.values()
                 if item.command.destination_ref == destination_ref),
                key=lambda item: item.command.publication_id,
            ))
        return values[:limit], len(values) <= MAX_PAGE


def _verify(authority, purpose: str, payload: bytes, tag: str, key_ref: str,
            code: PublicationCodeV1) -> None:
    try:
        valid = authority.verify(purpose, payload, tag, key_ref)
    except Exception:
        valid = False
    if valid is not True:
        raise _closed(code)


def _receipt_matches(receipt: AuthenticatedPublicationReceiptV1,
                     command: PublicationCommandV1, claim_digest: str,
                     ownership: TransferOwnershipV1) -> bool:
    return (
        receipt.publication_id == command.publication_id
        and receipt.command_digest == command.command_digest
        and receipt.run == command.run
        and receipt.source_identity_digest == command.source_identity_digest
        and receipt.destination_ref == command.destination_ref
        and receipt.destination_identity_digest == command.destination_identity_digest
        and receipt.mutation_id == command.mutation_id
        and receipt.claim_digest == claim_digest
        and receipt.ownership_id == ownership.ownership_id
        and receipt.authority_ref == command.destination_authority_ref
        and receipt.key_ref == command.destination_key_ref
    )


def _tombstone_matches(tombstone, command, claim_digest, ownership, permit):
    return (
        tombstone.publication_id == command.publication_id
        and tombstone.mutation_id == command.mutation_id
        and tombstone.command_digest == command.command_digest
        and tombstone.claim_digest == claim_digest
        and tombstone.destination_ref == command.destination_ref
        and tombstone.destination_identity_digest == command.destination_identity_digest
        and tombstone.destination_configuration_digest == command.destination_configuration_digest
        and tombstone.destination_policy_digest == command.destination_policy_digest
        and tombstone.destination_authority_ref == command.destination_authority_ref
        and tombstone.destination_key_ref == command.destination_key_ref
        and tombstone.fenced_ownership_id == ownership.ownership_id
        and tombstone.recovery_permit_id == permit.permit_id
        and tombstone.authority_ref == command.destination_authority_ref
        and tombstone.key_ref == command.destination_key_ref
    )


class PublicationOperationsV1:
    """Internal implementation of the unchanged staged ``ArtifactsOperations``."""

    def __init__(self, *, store: PublicationStorePortV1,
                 destinations: ArtifactDestinationRegistryPortV1,
                 sources: VerifiedArtifactSourcePortV1,
                 spool: ArtifactSpoolPortV1,
                 authority: EvidenceAuthorityPortV1,
                 clock) -> None:
        self._store = store
        self._destinations = destinations
        self._sources = sources
        self._spool = spool
        self._authority = authority
        self._clock = clock

    def _now(self) -> str:
        try:
            value = self._clock()
            return safe_ref(value, "timestamp")
        except Exception:
            raise _closed(PublicationCodeV1.EVIDENCE_INVALID) from None

    def _destination(self, destination_ref: str):
        try:
            resolved = self._destinations.resolve(destination_ref)
        except Exception:
            raise _closed(PublicationCodeV1.DESTINATION_MISSING) from None
        if (type(resolved) is not tuple or len(resolved) != 2
                or type(resolved[0]) is not AuthenticatedDestinationV1):
            raise _closed(PublicationCodeV1.DESTINATION_INVALID)
        descriptor, adapter = resolved
        if descriptor.destination_ref != destination_ref:
            raise _closed(PublicationCodeV1.DESTINATION_INVALID)
        _verify(self._authority, "publication-destination/v1", descriptor.payload,
                descriptor.tag, descriptor.key_ref,
                PublicationCodeV1.DESTINATION_INVALID)
        return descriptor, adapter

    def _source(self, run: TrainingRunRef) -> AuthenticatedVerifiedSourceV1:
        try:
            source = self._sources.describe(run)
            if type(source) is not AuthenticatedVerifiedSourceV1 or source.run != run:
                raise ValueError("source binding invalid")
            payload = source.payload
            source.source_identity_digest
        except Exception:
            raise _closed(PublicationCodeV1.SOURCE_INVALID) from None
        _verify(self._authority, "publication-verified-source/v1", payload,
                source.tag, source.key_ref, PublicationCodeV1.SOURCE_UNVERIFIED)
        return source

    @staticmethod
    def _validate_source_bounds(source, descriptor) -> None:
        try:
            artifacts = source.artifacts
            if (type(artifacts) is not tuple or not artifacts
                    or len(artifacts) > MAX_ARTIFACTS):
                raise ValueError("artifact count invalid")
            roles = tuple(item.role for item in artifacts)
            if roles != tuple(sorted(roles)) or len(roles) != len(set(roles)):
                raise ValueError("artifact roles invalid")
            total = 0
            for artifact in artifacts:
                if type(artifact) is not VerifiedArtifact:
                    raise TypeError("artifact type invalid")
                if (type(artifact.size_bytes) is not int
                        or artifact.size_bytes < 0
                        or artifact.size_bytes > MAX_ARTIFACT_BYTES
                        or artifact.size_bytes > descriptor.maximum_artifact_bytes):
                    raise ValueError("artifact size invalid")
                total += artifact.size_bytes
                if (total > MAX_ARTIFACT_BYTES
                        or total > descriptor.maximum_total_bytes):
                    raise ValueError("artifact total invalid")
        except Exception:
            raise _closed(PublicationCodeV1.SOURCE_INVALID) from None

    def _materialize(self, source: AuthenticatedVerifiedSourceV1,
                     publication_id: str) -> MaterializedSourceV1:
        spooled = []
        for expected in source.artifacts:
            sink = None
            try:
                stream = self._sources.open(RunArtifactRequest(
                    source.run, expected.role, expected.size_bytes or 1
                ))
                if (stream.run != source.run or stream.artifact != expected
                        or stream.maximum_bytes < expected.size_bytes):
                    raise ValueError("stream binding invalid")
                sink = self._spool.open(publication_id, expected.role,
                                        expected.size_bytes)
                total = 0
                digest = hashlib.sha256()
                for chunk in stream.iter_bytes():
                    if type(chunk) is not bytes or not chunk or len(chunk) > MAX_CHUNK_BYTES:
                        raise ValueError("chunk invalid")
                    total += len(chunk)
                    if total > expected.size_bytes:
                        raise ValueError("stream exceeds expected size")
                    digest.update(chunk)
                    sink.write(chunk)
                if total != expected.size_bytes or digest.hexdigest() != expected.sha256:
                    raise ValueError("stream failed EOF integrity")
                spool_ref = sink.finish()
                spooled.append(SpooledArtifactV1(expected, spool_ref))
            except Exception:
                if sink is not None:
                    try:
                        sink.abort()
                    except Exception:
                        pass
                raise _closed(PublicationCodeV1.SOURCE_CONTENT_INVALID) from None
        return MaterializedSourceV1(source.source_identity_digest, tuple(spooled))

    def _command(self, run: TrainingRunRef, descriptor, source):
        return PublicationCommandV1.build(
            run=run, source_identity_digest=source.source_identity_digest,
            source_inventory=source.artifacts,
            destination_ref=descriptor.destination_ref,
            destination_identity_digest=descriptor.identity_digest,
            destination_configuration_digest=descriptor.configuration_digest,
            destination_policy_digest=descriptor.policy_digest,
            maximum_artifact_bytes=descriptor.maximum_artifact_bytes,
            maximum_total_bytes=descriptor.maximum_total_bytes,
            destination_authority_ref=descriptor.authority_ref,
            destination_key_ref=descriptor.key_ref,
        )

    @staticmethod
    def _descriptor_matches_command(descriptor, command) -> bool:
        try:
            return (
                type(descriptor) is AuthenticatedDestinationV1
                and descriptor.destination_ref == command.destination_ref
                and descriptor.identity_digest == command.destination_identity_digest
                and descriptor.configuration_digest
                == command.destination_configuration_digest
                and descriptor.policy_digest == command.destination_policy_digest
                and descriptor.maximum_artifact_bytes
                == command.maximum_artifact_bytes
                and descriptor.maximum_total_bytes == command.maximum_total_bytes
                and descriptor.authority_ref == command.destination_authority_ref
                and descriptor.key_ref == command.destination_key_ref
            )
        except Exception:
            return False

    def _get(self, publication_id):
        try:
            record = self._store.get(publication_id)
        except Exception:
            raise _closed(PublicationCodeV1.STATE_CONFLICT) from None
        if record is not None and type(record) is not PublicationRecordV1:
            raise _closed(PublicationCodeV1.STATE_CONFLICT)
        return record

    def _verify_receipt(self, receipt, command, claim_digest, ownership):
        if type(receipt) is not AuthenticatedPublicationReceiptV1:
            raise _closed(PublicationCodeV1.EVIDENCE_INVALID)
        if (receipt.authority_ref != command.destination_authority_ref
                or receipt.key_ref != command.destination_key_ref):
            raise _closed(PublicationCodeV1.EVIDENCE_INVALID)
        _verify(self._authority, "publication-receipt/v1", receipt.payload,
                receipt.tag, receipt.key_ref, PublicationCodeV1.EVIDENCE_INVALID)
        if not _receipt_matches(receipt, command, claim_digest, ownership):
            raise _closed(PublicationCodeV1.EVIDENCE_INVALID)
        inventory = receipt.inventory
        if (inventory.authority_ref != command.destination_authority_ref
                or inventory.key_ref != command.destination_key_ref):
            raise _closed(PublicationCodeV1.EVIDENCE_INVALID)
        _verify(self._authority, "publication-destination-inventory/v1",
                inventory.payload, inventory.tag, inventory.key_ref,
                PublicationCodeV1.EVIDENCE_INVALID)
        if tuple((item.role, item.sha256, item.size_bytes)
                 for item in inventory.inventory.artifacts) != tuple(
                (item.role, item.sha256, item.size_bytes)
                for item in command.source_inventory):
            raise _closed(PublicationCodeV1.EVIDENCE_INVALID)

    def _readback(self, adapter, command, receipt):
        expected_by_role = {item.role: item for item in command.source_inventory}
        for destination_artifact in receipt.inventory.inventory.artifacts:
            expected = expected_by_role.get(destination_artifact.role)
            if (expected is None or destination_artifact.sha256 != expected.sha256
                    or destination_artifact.size_bytes != expected.size_bytes):
                raise _closed(PublicationCodeV1.EVIDENCE_INVALID)
            total = 0
            digest = hashlib.sha256()
            try:
                iterator = iter(adapter.iter_bytes(
                    command, destination_artifact,
                    destination_artifact.size_bytes or 1,
                ))
                for chunk in iterator:
                    if type(chunk) is not bytes or not chunk or len(chunk) > MAX_CHUNK_BYTES:
                        raise ValueError("readback chunk invalid")
                    total += len(chunk)
                    if total > destination_artifact.size_bytes:
                        raise ValueError("readback exceeds bound")
                    digest.update(chunk)
            except Exception:
                raise _closed(PublicationCodeV1.EVIDENCE_INVALID) from None
            if (total != destination_artifact.size_bytes
                    or digest.hexdigest() != destination_artifact.sha256):
                raise _closed(PublicationCodeV1.EVIDENCE_INVALID)

    def _lookup(self, record, adapter, permit):
        try:
            evidence = adapter.lookup(record.command, permit)
            if type(evidence) is not AuthenticatedLookupV1:
                raise ValueError("lookup type invalid")
            if (evidence.authority_ref != record.command.destination_authority_ref
                    or evidence.key_ref != record.command.destination_key_ref):
                return record
            _verify(self._authority, "publication-lookup/v1", evidence.payload,
                    evidence.tag, evidence.key_ref,
                    PublicationCodeV1.EVIDENCE_INVALID)
            ownership = record.ownership_history[-1]
            bound = (
                evidence.publication_id == record.command.publication_id
                and evidence.command_digest == record.command.command_digest
                and evidence.destination_identity_digest
                == record.command.destination_identity_digest
                and evidence.mutation_id == record.command.mutation_id
                and evidence.ownership_id == ownership.ownership_id
                and evidence.recovery_permit_id == permit.permit_id
            )
            if not bound:
                return self._finalize(permit, PublicationPhaseV1.CONFLICT,
                                      evidence)
            if evidence.outcome is LookupOutcomeV1.FOUND:
                receipt = evidence.receipt
                if type(receipt) is not AuthenticatedPublicationReceiptV1:
                    return record
                try:
                    self._verify_receipt(receipt, record.command,
                                         record.claim_digest, ownership)
                except PublicationErrorV1:
                    return self._finalize(permit, PublicationPhaseV1.CONFLICT,
                                          evidence)
                if record.receipt is not None and receipt != record.receipt:
                    return self._finalize(permit, PublicationPhaseV1.CONFLICT,
                                          evidence)
                self._readback(adapter, record.command, evidence.receipt)
                return self._finalize(permit, PublicationPhaseV1.VERIFIED,
                                      evidence,
                                      receipt=evidence.receipt)
            if evidence.outcome is LookupOutcomeV1.DEFINITELY_ABSENT:
                tombstone = evidence.tombstone
                if (type(tombstone) is not AuthenticatedPublicationTombstoneV1
                        or tombstone.authority_ref
                        != record.command.destination_authority_ref
                        or tombstone.key_ref != record.command.destination_key_ref):
                    return record
                _verify(self._authority, "publication-tombstone/v1",
                        tombstone.payload, tombstone.tag, tombstone.key_ref,
                        PublicationCodeV1.EVIDENCE_INVALID)
                if (not _tombstone_matches(
                        tombstone, record.command, record.claim_digest,
                        ownership, permit)
                        or tombstone.mutation_registry_digest
                        != evidence.mutation_registry_digest):
                    return self._finalize(permit, PublicationPhaseV1.CONFLICT,
                                          evidence)
                return self._finalize(permit, PublicationPhaseV1.ABSENT,
                                      evidence,
                                      tombstone=tombstone)
            if evidence.outcome is LookupOutcomeV1.CONFLICT:
                return self._finalize(permit, PublicationPhaseV1.CONFLICT,
                                      evidence)
            return record
        except PublicationErrorV1:
            return record
        except Exception:
            return record

    def _finalize(self, permit, phase, outcome, *, receipt=None,
                  tombstone=None):
        try:
            result = self._store.finalize_recovery(
                permit, phase, self._now(), outcome, receipt=receipt,
                tombstone=tombstone,
            )
            if type(result) is not PublicationRecordV1:
                raise ValueError("store result invalid")
            return result
        except Exception:
            return self._get(permit.publication_id)

    def _recover(self, record, descriptor, adapter):
        if not self._descriptor_matches_command(descriptor, record.command):
            return record
        try:
            decision = self._store.recover_transfer(
                record.command.publication_id, record.command.command_digest,
                self._now(),
            )
        except Exception:
            return self._get(record.command.publication_id)
        if type(decision) is not RecoveryDecisionV1:
            return self._get(record.command.publication_id)
        if decision.disposition is RecoveryDispositionV1.PERMITTED:
            return self._lookup(decision.record, adapter, decision.permit)
        return decision.record

    def _project(self, record):
        mapping = {
            PublicationPhaseV1.CLAIMED: PublicationState.CLAIMED,
            PublicationPhaseV1.TRANSFERRING: PublicationState.TRANSFERRING,
            PublicationPhaseV1.COMMITTED: PublicationState.COMMITTED,
            PublicationPhaseV1.VERIFIED: PublicationState.VERIFIED,
            PublicationPhaseV1.AMBIGUOUS: PublicationState.AMBIGUOUS,
            PublicationPhaseV1.ABSENT: PublicationState.AMBIGUOUS,
            PublicationPhaseV1.CONFLICT: PublicationState.AMBIGUOUS,
            PublicationPhaseV1.FAILED_BEFORE_EFFECT: PublicationState.FAILED_BEFORE_EFFECT,
        }
        artifacts = record.command.source_inventory if record.phase is PublicationPhaseV1.VERIFIED else ()
        return PublicationResult(
            "synaptic-publication-result/v1",
            PublicationRef(record.command.publication_id,
                           record.command.destination_ref),
            TrainingRunRef.from_dict(record.command.run.to_dict()),
            mapping[record.phase], artifacts,
        )

    def destinations(self) -> DestinationPage:
        try:
            descriptors, complete = self._destinations.list(MAX_PAGE + 1)
        except Exception:
            raise _closed(PublicationCodeV1.PAGE_INCOMPLETE) from None
        if type(descriptors) is not tuple or complete is not True or len(descriptors) > MAX_PAGE:
            raise _closed(PublicationCodeV1.PAGE_INCOMPLETE)
        result = []
        refs = set()
        for descriptor in descriptors:
            if type(descriptor) is not AuthenticatedDestinationV1 or descriptor.destination_ref in refs:
                raise _closed(PublicationCodeV1.DESTINATION_INVALID)
            _verify(self._authority, "publication-destination/v1", descriptor.payload,
                    descriptor.tag, descriptor.key_ref,
                    PublicationCodeV1.DESTINATION_INVALID)
            refs.add(descriptor.destination_ref)
            result.append(ArtifactDestination(descriptor.destination_ref,
                                              descriptor.display_name))
        return DestinationPage(tuple(sorted(result, key=lambda item: item.destination_ref)))

    def publications(self, destination_ref: str) -> PublicationPage:
        safe_ref(destination_ref, "destination_ref")
        try:
            records, complete = self._store.list(destination_ref, MAX_PAGE + 1)
        except PublicationErrorV1:
            raise
        except Exception:
            raise _closed(PublicationCodeV1.PAGE_INCOMPLETE) from None
        if type(records) is not tuple or complete is not True or len(records) > MAX_PAGE:
            raise _closed(PublicationCodeV1.PAGE_INCOMPLETE)
        if any(type(item) is not PublicationRecordV1 for item in records):
            raise _closed(PublicationCodeV1.PAGE_INCOMPLETE)
        publication_ids = tuple(item.command.publication_id for item in records)
        if (any(item.command.destination_ref != destination_ref for item in records)
                or len(publication_ids) != len(set(publication_ids))
                or publication_ids != tuple(sorted(publication_ids))):
            raise _closed(PublicationCodeV1.PAGE_INCOMPLETE)
        return PublicationPage(tuple(self._project(item) for item in records))

    def publish(self, request: PublicationRequest) -> PublicationResult:
        if type(request) is not PublicationRequest:
            raise TypeError("request must be exact PublicationRequest")
        descriptor, adapter = self._destination(request.destination_ref)
        source = self._source(request.run)
        self._validate_source_bounds(source, descriptor)
        try:
            command = self._command(request.run, descriptor, source)
        except Exception:
            raise _closed(PublicationCodeV1.SOURCE_INVALID) from None
        existing = self._get(command.publication_id)
        if existing is not None:
            if existing.command.command_digest != command.command_digest:
                raise _closed(PublicationCodeV1.PUBLICATION_CONFLICT)
            if existing.phase in (PublicationPhaseV1.TRANSFERRING,
                                  PublicationPhaseV1.COMMITTED,
                                  PublicationPhaseV1.AMBIGUOUS):
                existing = self._recover(existing, descriptor, adapter)
                return self._project(existing)
            if existing.phase is not PublicationPhaseV1.CLAIMED:
                return self._project(existing)
            claimed = existing
        else:
            try:
                claimed, created = self._store.claim(
                    PublicationRecordV1.claim(command, self._now()))
            except Exception:
                raise _closed(PublicationCodeV1.STATE_CONFLICT) from None
            if type(claimed) is not PublicationRecordV1 or type(created) is not bool:
                raise _closed(PublicationCodeV1.STATE_CONFLICT)
            if not created and claimed.phase is not PublicationPhaseV1.CLAIMED:
                return self._project(claimed)
        try:
            materialized = self._materialize(source, command.publication_id)
        except PublicationErrorV1:
            descendant = claimed.transition(
                PublicationPhaseV1.FAILED_BEFORE_EFFECT, self._now())
            try:
                self._store.compare_and_swap(claimed.record_digest, descendant)
            except Exception:
                pass
            raise
        try:
            admission = self._store.begin_transfer(
                command.publication_id, claimed.record_digest, self._now())
        except Exception:
            return self._project(self._get(command.publication_id))
        if type(admission) is not TransferAdmissionV1:
            return self._project(self._get(command.publication_id))
        if (admission.disposition is not TransferDispositionV1.ACQUIRED
                or type(admission.ownership) is not TransferOwnershipV1):
            return self._project(admission.record)
        ownership = admission.ownership
        try:
            receipt = adapter.publish_once(command, materialized, ownership)
            self._verify_receipt(receipt, command, claimed.claim_digest,
                                 ownership)
        except Exception:
            try:
                decision = self._store.relinquish_uncertain(
                    ownership, self._now())
                current = decision.record
            except Exception:
                current = self._get(command.publication_id)
            return self._project(current)
        try:
            committed = self._store.complete_transfer(
                ownership, receipt, False, self._now())
        except Exception:
            try:
                decision = self._store.relinquish_uncertain(
                    ownership, self._now())
                current = decision.record
            except Exception:
                current = self._get(command.publication_id)
            if current.phase is PublicationPhaseV1.AMBIGUOUS:
                current = self._recover(current, descriptor, adapter)
            return self._project(current)
        try:
            self._readback(adapter, command, receipt)
        except Exception:
            try:
                current = self._store.relinquish_uncertain(
                    ownership, self._now()).record
            except Exception:
                current = self._get(command.publication_id)
            return self._project(current)
        try:
            final = self._store.complete_transfer(
                ownership, receipt, True, self._now())
        except Exception:
            current = self._get(command.publication_id)
            if current.phase in (PublicationPhaseV1.COMMITTED,
                                 PublicationPhaseV1.AMBIGUOUS):
                current = self._recover(current, descriptor, adapter)
            return self._project(current)
        return self._project(final)

    def verify(self, publication: PublicationRef) -> PublicationVerification:
        if type(publication) is not PublicationRef:
            raise TypeError("publication must be exact PublicationRef")
        record = self._get(publication.publication_id)
        if record is None or record.command.destination_ref != publication.destination_ref:
            raise _closed(PublicationCodeV1.PUBLICATION_MISSING)
        if record.phase in (PublicationPhaseV1.TRANSFERRING,
                            PublicationPhaseV1.COMMITTED,
                            PublicationPhaseV1.AMBIGUOUS):
            descriptor, adapter = self._destination(record.command.destination_ref)
            record = self._recover(record, descriptor, adapter)
        return PublicationVerification(
            publication, record.phase is PublicationPhaseV1.VERIFIED, self._now()
        )


__all__ = [
    "ArtifactDestinationRegistryPortV1", "ArtifactSpoolPortV1",
    "AuthenticatedDestinationInventoryV1", "AuthenticatedDestinationV1",
    "AuthenticatedLookupV1", "AuthenticatedPublicationReceiptV1",
    "AuthenticatedPublicationTombstoneV1", "AuthenticatedVerifiedSourceV1",
    "DestinationArtifactV1", "DestinationInventoryV1",
    "DestinationPublicationPortV1", "EvidenceAuthorityPortV1", "LookupOutcomeV1",
    "LookupRecoveryPermitV1", "MaterializedSourceV1", "PublicationCodeV1",
    "PublicationErrorV1", "PublicationEventKindV1", "PublicationEventV1",
    "PublicationOperationsV1", "PublicationPhaseV1", "PublicationRecordV1",
    "PublicationStorePortV1", "PublicationTransitionKernelV1",
    "RecoveryDecisionV1", "RecoveryDispositionV1",
    "SpooledArtifactV1", "SpoolSinkPortV1",
    "StrongInMemoryPublicationStoreV1", "TransferAdmissionV1",
    "TransferDispositionV1", "TransferOwnershipV1",
    "VerifiedArtifactSourcePortV1",
]
