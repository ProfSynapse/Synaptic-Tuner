"""Sole mutation admission boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ._effect_executor import _ProviderEffectExecutor
from .contracts import (
    EffectCollision, EffectDisposition, EffectObservation, EffectState,
    LifecycleRepository, _AttemptDisposition, digest,
)
from .operation import OperationBindingV1


@dataclass(frozen=True, slots=True)
class MutationCommandV1:
    operation: OperationBindingV1
    bundle_digest: str
    stage_claim_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.operation, OperationBindingV1):
            raise TypeError("operation must be OperationBindingV1")
        object.__setattr__(self, "bundle_digest", digest(self.bundle_digest, "bundle_digest"))
        object.__setattr__(
            self, "stage_claim_digest", digest(self.stage_claim_digest, "stage_claim_digest")
        )

    def __getattr__(self, name: str):
        if name in self.operation.__dataclass_fields__ or name in {
            "source_digest", "artifact_slot_ref", "allowed_secret_refs_digest",
        }:
            return getattr(self.operation, name)
        raise AttributeError(name)

    @property
    def operation_binding_digest(self) -> str:
        return self.operation.digest

    @property
    def canonical_bytes(self) -> bytes:
        document = {
            "schema_version": "synaptic-mutation-command/v1",
            "operation_binding": self.operation.to_dict(),
            "operation_binding_digest": self.operation.digest,
            "bundle_digest": self.bundle_digest,
            "stage_claim_digest": self.stage_claim_digest,
        }
        return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @property
    def digest(self) -> str:
        return hashlib.sha256(b"synaptic-mutation-command/v1\0" + self.canonical_bytes).hexdigest()

    @classmethod
    def from_stage(cls, operation: OperationBindingV1, receipt) -> "MutationCommandV1":
        from .providers.modal.contracts import StageReceiptV1

        if not isinstance(operation, OperationBindingV1) or not isinstance(receipt, StageReceiptV1):
            raise TypeError("operation and stage receipt must be canonical values")
        target = operation.stage_target
        if (
            receipt.effect_id != operation.effect.effect_id
            or receipt.operation_binding_digest != operation.digest
            or receipt.control_volume_id != target.control_volume_id
            or receipt.artifact_volume_id != target.artifact_volume_id
        ):
            raise ValueError("stage receipt does not bind the operation")
        return cls(operation, receipt.bundle_digest, receipt.claim_digest)

    @classmethod
    def from_bytes(cls, value: bytes) -> "MutationCommandV1":
        """Parse the one canonical payload accepted by provider mutators/workers."""
        if not isinstance(value, bytes) or not value:
            raise ValueError("mutation command must be nonempty canonical JSON bytes")
        try:
            document = json.loads(value.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("mutation command must be canonical JSON") from exc
        expected = {
            "schema_version", "operation_binding", "operation_binding_digest",
            "bundle_digest", "stage_claim_digest",
        }
        if not isinstance(document, dict) or set(document) != expected:
            raise ValueError("mutation command contains missing or unknown fields")
        if document.get("schema_version") != "synaptic-mutation-command/v1":
            raise ValueError("unsupported mutation-command schema")
        operation = OperationBindingV1.from_dict(document["operation_binding"])
        if document["operation_binding_digest"] != operation.digest:
            raise ValueError("mutation command operation digest mismatch")
        result = cls(
            operation=operation,
            bundle_digest=document["bundle_digest"],
            stage_claim_digest=document["stage_claim_digest"],
        )
        if result.canonical_bytes != value:
            raise ValueError("mutation command is not canonical")
        return result


class MutationBroker:
    def __init__(self, repository: LifecycleRepository, executor: _ProviderEffectExecutor):
        self._repository = repository
        self._executor = executor

    def execute(self, command: MutationCommandV1, *, expected_revision: int) -> EffectObservation:
        if not isinstance(command, MutationCommandV1):
            raise TypeError("command must be MutationCommandV1")
        admission = self._repository.compare_and_consume_attempt(
            command.project_ref, command.run_id, expected_revision=expected_revision,
            grant_ref=command.grant_ref, canonical_command=command,
        )
        if admission.disposition is _AttemptDisposition.LOOKUP_ONLY:
            effect = admission.effect
            if effect.state is EffectState.FOUND:
                return EffectObservation(
                    effect.identity, EffectDisposition.FOUND,
                    effect.provider_job_ref, effect.receipt_digest,
                )
            if effect.state is EffectState.DEFINITELY_ABSENT:
                return EffectObservation(effect.identity, EffectDisposition.DEFINITELY_ABSENT)
            return EffectObservation(effect.identity, EffectDisposition.INDETERMINATE)
        observation = self._executor.execute_once(command.canonical_bytes)
        if observation.identity != command.effect:
            raise EffectCollision("provider outcome effect mismatch")
        self._repository.record_attempt_outcome(
            command.project_ref, command.run_id, expected_revision=admission.record.revision,
            command_digest=command.digest, observation=observation,
        )
        return observation


__all__ = ["MutationBroker", "MutationCommandV1"]
