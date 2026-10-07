"""Explicit Docker v1 same-process composition contracts.

This module does not promise restart or cross-process durability. Hosts that need
those guarantees must supply a different composition rather than infer them from
this in-memory runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from threading import RLock
from typing import Protocol

from synaptic_tuner.api.v1.planning import ProviderPlanContextV1, TrainingPlan
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.training_facade import TrainingPreflight
from tuner.execution.providers.docker_provider_v1.model import (
    AuthenticatedDockerCommandBindingV1,
    DockerCommandBindingV1,
    DockerEffectIdentityV1,
    DockerLabelsV1,
    DockerProfileV1,
    PreparedDockerPlanV1,
    validated_profile_snapshot,
)
from tuner.execution.providers.docker_provider_v1.ports import (
    DockerCommandBindingAuthorityPortV1,
    DockerControlPortV1,
    DockerEvidenceAuthorityPortV1,
    DockerImageInventoryPortV1,
    DockerSourceSealPortV1,
)
from tuner.execution.coordinator_v1.model import WorkflowRecordV1
from tuner.execution.foundation_v2.canonical import digest_text


def _labels_snapshot(value: DockerLabelsV1 | None) -> DockerLabelsV1 | None:
    if value is None:
        return None
    if type(value) is not DockerLabelsV1:
        raise TypeError("exact Docker labels required")
    return DockerLabelsV1(**{field.name: getattr(value, field.name) for field in fields(DockerLabelsV1)})


def _binding_snapshot(
    value: AuthenticatedDockerCommandBindingV1,
) -> AuthenticatedDockerCommandBindingV1:
    if type(value) is not AuthenticatedDockerCommandBindingV1:
        raise TypeError("exact authenticated Docker binding required")
    content = value.content
    if type(content) is not DockerCommandBindingV1:
        raise TypeError("exact Docker binding content required")
    identity = content.identity
    if type(identity) is not DockerEffectIdentityV1 or type(identity.plan) is not PreparedDockerPlanV1:
        raise TypeError("exact Docker binding identity required")
    plan = identity.plan
    plan = PreparedDockerPlanV1(
        validated_profile_snapshot(plan.profile), plan.project_ref, plan.run_id,
        plan.plan_fingerprint, plan.source_digest, plan.preparation_digest,
    )
    identity = DockerEffectIdentityV1(
        identity.command_digest, identity.effect_id, identity.effect_kind, plan,
    )
    rebuilt_content = DockerCommandBindingV1(
        identity, bytes(content.command_bytes),
        None if content.original_submit_command_bytes is None
        else bytes(content.original_submit_command_bytes),
        content.cancel_container_ref, content.cancel_reason_digest,
        _labels_snapshot(content.cancel_submit_labels),
        content.cancel_authorization_digest,
    )
    rebuilt = AuthenticatedDockerCommandBindingV1(
        rebuilt_content, value.binding_digest, value.authority_ref,
        value.key_ref, value.tag,
    )
    if rebuilt != value:
        raise ValueError("Docker binding reconstruction mismatch")
    return rebuilt


class DockerBindingAuthorityV1(DockerCommandBindingAuthorityPortV1, Protocol):
    authority_ref: str
    key_ref: str

    def issue(
        self, binding: DockerCommandBindingV1
    ) -> AuthenticatedDockerCommandBindingV1: ...

    def authenticate(self, value: AuthenticatedDockerCommandBindingV1) -> bool: ...


class DockerSameProcessRuntimeV1(Protocol):
    def start(self) -> WorkflowRecordV1: ...
    def reconcile(self) -> WorkflowRecordV1: ...
    def binding(self, effect_kind: str) -> AuthenticatedDockerCommandBindingV1: ...


class DockerSameProcessBindingStoreV1:
    """Authenticated exact binding store with same-process scope only."""

    __slots__ = ("authority_ref", "key_ref", "_authority", "_lock", "_values")

    def __init__(self, binding_authority: DockerBindingAuthorityV1):
        authority_ref = getattr(binding_authority, "authority_ref", None)
        key_ref = getattr(binding_authority, "key_ref", None)
        if type(authority_ref) is not str or type(key_ref) is not str:
            raise TypeError("binding authority identity required")
        self.authority_ref = authority_ref
        self.key_ref = key_ref
        self._authority = binding_authority
        self._lock = RLock()
        self._values: dict[str, AuthenticatedDockerCommandBindingV1] = {}

    def _validated(self, value: AuthenticatedDockerCommandBindingV1):
        candidate = _binding_snapshot(value)
        probe = _binding_snapshot(candidate)
        try:
            authenticated = self._authority.authenticate(probe)
            probe = _binding_snapshot(probe)
        except Exception:
            authenticated = False
        if (candidate.authority_ref != self.authority_ref
                or candidate.key_ref != self.key_ref
                or authenticated is not True
                or probe != candidate):
            raise ValueError("Docker binding authentication failed")
        return candidate

    def publish_once(
        self, value: AuthenticatedDockerCommandBindingV1
    ) -> AuthenticatedDockerCommandBindingV1:
        candidate = self._validated(value)
        key = candidate.content.command_digest
        with self._lock:
            existing = self._values.get(key)
            if existing is None:
                self._values[key] = candidate
                return _binding_snapshot(candidate)
            retained = self._validated(existing)
            if retained != candidate:
                raise ValueError("Docker command binding conflict")
            return _binding_snapshot(retained)

    def resolve(self, command_digest: str) -> AuthenticatedDockerCommandBindingV1:
        digest_text(command_digest, "command_digest")
        with self._lock:
            value = self._values.get(command_digest)
            if value is None:
                raise KeyError("Docker command binding missing")
            retained = self._validated(value)
            if retained.content.command_digest != command_digest:
                raise ValueError("Docker command binding key mismatch")
            return _binding_snapshot(retained)


@dataclass(frozen=True, slots=True)
class DockerSameProcessLaunchV1:
    profile: DockerProfileV1
    context: ProviderPlanContextV1
    plan: TrainingPlan
    run: TrainingRunRef
    preflight: TrainingPreflight

    def __post_init__(self) -> None:
        if (type(self.profile) is not DockerProfileV1
                or type(self.context) is not ProviderPlanContextV1
                or type(self.plan) is not TrainingPlan
                or type(self.run) is not TrainingRunRef
                or type(self.preflight) is not TrainingPreflight):
            raise TypeError("exact Docker same-process launch values required")
        profile = validated_profile_snapshot(self.profile)
        context = ProviderPlanContextV1.from_dict(self.context.to_dict())
        plan = TrainingPlan.from_dict(self.plan.to_dict())
        run = TrainingRunRef.from_dict(self.run.to_dict())
        preflight = TrainingPreflight.from_dict(self.preflight.to_dict())
        if (
            context.provider != profile.provider
            or context.basis_digest != plan.basis.basis_digest
            or context.descriptor_digest != profile.descriptor.descriptor_digest
            or context.profile_digest != profile.profile_digest
            or plan.provider_plan.context_digest != context.provider_context_digest
            or run.project_ref != plan.basis.project_ref
            or plan.basis.workload_digest != profile.workload.workload_digest
            or plan.basis.runtime_digest != profile.runtime.digest
            or plan.basis.artifact_policy_digest != profile.artifacts.digest
            or not preflight.binds(plan)
        ):
            raise ValueError("Docker launch bindings differ")
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "run", run)
        object.__setattr__(self, "preflight", preflight)


def _requires(value: object, *methods: str) -> None:
    if any(not callable(getattr(value, method, None)) for method in methods):
        raise TypeError("Docker host port is incomplete")


@dataclass(frozen=True, slots=True)
class DockerCoordinatorHostPortsV1:
    binding_store: DockerSameProcessBindingStoreV1
    binding_authority: DockerBindingAuthorityV1
    image_inventory: DockerImageInventoryPortV1
    source_seals: DockerSourceSealPortV1
    control: DockerControlPortV1
    evidence_authority: DockerEvidenceAuthorityPortV1

    def __post_init__(self) -> None:
        if type(self.binding_store) is not DockerSameProcessBindingStoreV1:
            raise TypeError("exact same-process binding store required")
        _requires(self.binding_authority, "issue", "authenticate")
        _requires(self.image_inventory, "require_present")
        _requires(self.source_seals, "seal_read_only", "lookup")
        _requires(self.control, "create_once", "start_once", "lookup")
        _requires(
            self.evidence_authority,
            "authenticate_source_seal", "authenticate_absence",
        )
        if (
            self.binding_store.authority_ref
            != getattr(self.binding_authority, "authority_ref", None)
            or self.binding_store.key_ref
            != getattr(self.binding_authority, "key_ref", None)
        ):
            raise ValueError("binding store and authority differ")


def compose_docker_same_process_coordinator_v1(
    launch: DockerSameProcessLaunchV1,
    ports: DockerCoordinatorHostPortsV1,
) -> DockerSameProcessRuntimeV1:
    from tuner.execution.providers.docker_provider_v1.composition import (
        compose_docker_same_process_coordinator_v1 as compose,
    )

    return compose(launch, ports)


__all__ = [
    "DockerBindingAuthorityV1",
    "DockerCoordinatorHostPortsV1",
    "DockerSameProcessBindingStoreV1",
    "DockerSameProcessLaunchV1",
    "DockerSameProcessRuntimeV1",
    "compose_docker_same_process_coordinator_v1",
]
