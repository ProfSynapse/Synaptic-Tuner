"""Typed pre-stage operation identity.

The operation digest is derived once from canonical predecessor documents and a
typed stage target.  Staging and final mutation consume this value; callers do
not supply an operation digest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping

from .contracts import EffectIdentity, EffectKind, digest, safe_ref


_PREDECESSORS = frozenset({
    "artifact-contract.json", "deployment.json", "execution-source.json",
    "invocation-intent.json", "log-terminal-policy.json", "plan.json",
    "workload.json",
})


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _load_canonical(value: bytes, name: str) -> dict[str, object]:
    if not isinstance(value, bytes) or not value:
        raise ValueError(f"{name} must be nonempty canonical JSON bytes")
    try:
        document = json.loads(value.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be canonical JSON") from exc
    if not isinstance(document, dict) or _canonical(document) != value:
        raise ValueError(f"{name} must be a canonical JSON object")
    return document


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True, slots=True)
class ModalStageTargetV1:
    artifact_slot_ref: str
    control_volume_id: str
    artifact_volume_id: str
    output_prefix: str
    generation: int
    key_ref: str

    def __post_init__(self) -> None:
        for name in (
            "artifact_slot_ref", "control_volume_id", "artifact_volume_id",
            "output_prefix", "key_ref",
        ):
            object.__setattr__(self, name, safe_ref(getattr(self, name), name))
        if self.control_volume_id == self.artifact_volume_id:
            raise ValueError("control and artifact volumes must be distinct")
        if type(self.generation) is not int or not 1 <= self.generation <= 2**31 - 1:
            raise ValueError("generation must be a bounded exact integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_slot_ref": self.artifact_slot_ref,
            "control_volume_id": self.control_volume_id,
            "artifact_volume_id": self.artifact_volume_id,
            "output_prefix": self.output_prefix,
            "generation": self.generation,
            "key_ref": self.key_ref,
        }


@dataclass(frozen=True, slots=True)
class OperationBindingV1:
    project_ref: str
    run_id: str
    effect: EffectIdentity
    grant_ref: str
    plan_fingerprint: str
    execution_source_digest: str
    workload_digest: str
    deployment_attestation_digest: str
    artifact_contract_digest: str
    log_policy_digest: str
    invocation_intent_digest: str
    resource_digest: str
    quote_digest: str
    secret_requirements_digest: str
    invocation_arguments_digest: str
    invocation_nonce: str
    stage_target: ModalStageTargetV1
    target_provider_job_ref: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.effect, EffectIdentity):
            raise TypeError("effect must be EffectIdentity")
        if not isinstance(self.stage_target, ModalStageTargetV1):
            raise TypeError("stage_target must be ModalStageTargetV1")
        for name in ("project_ref", "run_id", "grant_ref", "invocation_nonce"):
            object.__setattr__(self, name, safe_ref(getattr(self, name), name))
        for name in (
            "plan_fingerprint", "execution_source_digest", "workload_digest",
            "deployment_attestation_digest", "artifact_contract_digest",
            "log_policy_digest", "invocation_intent_digest", "resource_digest",
            "quote_digest", "secret_requirements_digest", "invocation_arguments_digest",
        ):
            object.__setattr__(self, name, digest(getattr(self, name), name))
        if self.effect.kind is EffectKind.CANCEL:
            if self.target_provider_job_ref is None:
                raise ValueError("cancel requires an exact provider job")
            object.__setattr__(
                self, "target_provider_job_ref",
                safe_ref(self.target_provider_job_ref, "target_provider_job_ref"),
            )
        elif self.target_provider_job_ref is not None:
            raise ValueError("submit cannot target a provider job")
        from .providers.modal.contracts import operation_path
        if self.stage_target.output_prefix != operation_path(self.effect.effect_id, "output"):
            raise ValueError("stage output prefix must be derived from the effect identity")

    @property
    def source_digest(self) -> str:
        return self.execution_source_digest

    @property
    def artifact_slot_ref(self) -> str:
        return self.stage_target.artifact_slot_ref

    @property
    def allowed_secret_refs_digest(self) -> str:
        return self.secret_requirements_digest

    def to_dict(self) -> dict[str, object]:
        scope = self.effect.scope
        return {
            "schema_version": "synaptic-operation-binding/v1",
            "project_ref": self.project_ref, "run_id": self.run_id,
            "effect": {
                "effect_id": self.effect.effect_id, "effect_key": self.effect.effect_key,
                "kind": self.effect.kind.value,
                "scope": {"provider": scope.provider, "account_ref": scope.account_ref,
                          "namespace_ref": scope.namespace_ref},
            },
            "grant_ref": self.grant_ref, "plan_fingerprint": self.plan_fingerprint,
            "execution_source_digest": self.execution_source_digest,
            "workload_digest": self.workload_digest,
            "deployment_attestation_digest": self.deployment_attestation_digest,
            "artifact_contract_digest": self.artifact_contract_digest,
            "log_policy_digest": self.log_policy_digest,
            "invocation_intent_digest": self.invocation_intent_digest,
            "resource_digest": self.resource_digest, "quote_digest": self.quote_digest,
            "secret_requirements_digest": self.secret_requirements_digest,
            "invocation_arguments_digest": self.invocation_arguments_digest,
            "invocation_nonce": self.invocation_nonce,
            "stage_target": self.stage_target.to_dict(),
            "target_provider_job_ref": self.target_provider_job_ref,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical(self.to_dict())

    @property
    def digest(self) -> str:
        return hashlib.sha256(b"synaptic-operation-binding/v1\0" + self.canonical_bytes).hexdigest()

    @classmethod
    def from_predecessors(
        cls, *, project_ref: str, grant_ref: str, effect: EffectIdentity,
        stage_target: ModalStageTargetV1, member_documents: Mapping[str, bytes],
        target_provider_job_ref: str | None = None,
    ) -> "OperationBindingV1":
        if not isinstance(member_documents, Mapping) or set(member_documents) != _PREDECESSORS:
            raise ValueError("operation binding requires the exact seven predecessor members")
        docs = {name: _load_canonical(member_documents[name], name) for name in _PREDECESSORS}
        plan = docs["plan.json"]
        invocation = docs["invocation-intent.json"]
        deployment = docs["deployment.json"]
        source = docs["execution-source.json"]
        if (
            plan.get("effect_id") != effect.effect_id
            or plan.get("effect_key") != effect.effect_key
            or plan.get("provider") != effect.scope.provider
            or plan.get("account_ref") != effect.scope.account_ref
            or plan.get("namespace_ref") != effect.scope.namespace_ref
            or plan.get("artifact_slot_ref") != stage_target.artifact_slot_ref
            or invocation.get("invocation_nonce") is None
            or invocation.get("run_id") != plan.get("run_id")
            or source.get("run_id") != plan.get("run_id")
        ):
            raise ValueError("predecessor members do not bind one operation identity")
        arguments = {
            "interpreter": invocation.get("interpreter"), "argv": invocation.get("argv"),
            "cwd": invocation.get("cwd"),
            "environment_digest": invocation.get("environment_digest"),
        }
        return cls(
            project_ref=project_ref, run_id=plan["run_id"], effect=effect,
            grant_ref=grant_ref, plan_fingerprint=_sha(member_documents["plan.json"]),
            execution_source_digest=_sha(member_documents["execution-source.json"]),
            workload_digest=_sha(member_documents["workload.json"]),
            deployment_attestation_digest=deployment["attestation_digest"],
            artifact_contract_digest=_sha(member_documents["artifact-contract.json"]),
            log_policy_digest=_sha(member_documents["log-terminal-policy.json"]),
            invocation_intent_digest=_sha(member_documents["invocation-intent.json"]),
            resource_digest=plan["resource_digest"], quote_digest=plan["quote_digest"],
            secret_requirements_digest=plan["secret_requirements_digest"],
            invocation_arguments_digest=_sha(_canonical(arguments)),
            invocation_nonce=invocation["invocation_nonce"], stage_target=stage_target,
            target_provider_job_ref=target_provider_job_ref,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "OperationBindingV1":
        expected = {
            "schema_version", "project_ref", "run_id", "effect", "grant_ref",
            "plan_fingerprint", "execution_source_digest", "workload_digest",
            "deployment_attestation_digest", "artifact_contract_digest",
            "log_policy_digest", "invocation_intent_digest", "resource_digest",
            "quote_digest", "secret_requirements_digest", "invocation_arguments_digest",
            "invocation_nonce", "stage_target", "target_provider_job_ref",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("operation binding contains missing or unknown fields")
        if value.get("schema_version") != "synaptic-operation-binding/v1":
            raise ValueError("unsupported operation-binding schema")
        raw_effect = value.get("effect")
        raw_target = value.get("stage_target")
        if not isinstance(raw_effect, Mapping) or set(raw_effect) != {
            "effect_id", "effect_key", "kind", "scope",
        }:
            raise ValueError("operation effect is malformed")
        raw_scope = raw_effect.get("scope")
        if not isinstance(raw_scope, Mapping) or set(raw_scope) != {
            "provider", "account_ref", "namespace_ref",
        }:
            raise ValueError("operation scope is malformed")
        if not isinstance(raw_target, Mapping) or set(raw_target) != {
            "artifact_slot_ref", "control_volume_id", "artifact_volume_id",
            "output_prefix", "generation", "key_ref",
        }:
            raise ValueError("operation stage target is malformed")
        from .contracts import ExecutionScope
        result = cls(
            project_ref=value["project_ref"], run_id=value["run_id"],
            effect=EffectIdentity(
                raw_effect["effect_id"], raw_effect["effect_key"],
                EffectKind(raw_effect["kind"]), ExecutionScope(**raw_scope),
            ),
            grant_ref=value["grant_ref"], plan_fingerprint=value["plan_fingerprint"],
            execution_source_digest=value["execution_source_digest"],
            workload_digest=value["workload_digest"],
            deployment_attestation_digest=value["deployment_attestation_digest"],
            artifact_contract_digest=value["artifact_contract_digest"],
            log_policy_digest=value["log_policy_digest"],
            invocation_intent_digest=value["invocation_intent_digest"],
            resource_digest=value["resource_digest"], quote_digest=value["quote_digest"],
            secret_requirements_digest=value["secret_requirements_digest"],
            invocation_arguments_digest=value["invocation_arguments_digest"],
            invocation_nonce=value["invocation_nonce"],
            stage_target=ModalStageTargetV1(**raw_target),
            target_provider_job_ref=value["target_provider_job_ref"],
        )
        if result.to_dict() != dict(value):
            raise ValueError("operation binding is not canonical")
        return result


__all__ = ["ModalStageTargetV1", "OperationBindingV1"]
