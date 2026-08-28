"""Canonical, provider-local values for the hermetic Docker v1 adapter."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

from synaptic_tuner.api.v1.providers import ProviderCapabilities, ProviderDescriptor, ProviderRef
from synaptic_tuner.api.v1.results import VerifiedArtifact

from ...foundation_v2.canonical import canonical_bytes, digest_text, domain_digest, safe_ref
from ...foundation_v2.executors import AdapterDescriptorV1, ExecutorDescriptorV1
from ...foundation_v2.references import ExecutionScopeV1


MAX_ARGUMENTS = 64
MAX_ARGUMENT_BYTES = 32_768
MAX_ENVIRONMENT_KEYS = 64
MAX_ARTIFACTS = 256
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024 * 1024
MAX_ARTIFACT_TOTAL_BYTES = 64 * 1024 * 1024 * 1024
MAX_LOG_ENTRIES = 200
MAX_LOG_BYTES = 262_144
MAX_EVIDENCE_BYTES = 65_536
_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


class DockerDiagnosticCodeV1(str, Enum):
    BINDING_MISMATCH = "docker_binding_mismatch"
    INVALID_PLAN = "docker_invalid_plan"
    IMAGE_UNAVAILABLE = "docker_image_unavailable"
    SOURCE_UNSEALED = "docker_source_unsealed"
    SOURCE_WRITABLE = "docker_source_writable"
    CREATE_COLLISION = "docker_create_collision"
    CREATE_INDETERMINATE = "docker_create_indeterminate"
    START_INDETERMINATE = "docker_start_indeterminate"
    STOP_INDETERMINATE = "docker_stop_indeterminate"
    LOOKUP_INDETERMINATE = "docker_lookup_indeterminate"
    MALFORMED_EVIDENCE = "docker_malformed_evidence"
    BOUNDS_EXCEEDED = "docker_bounds_exceeded"
    AUTHENTICATION_FAILED = "docker_authentication_failed"


class DockerProviderError(RuntimeError):
    """Closed error which never includes provider exception text."""

    def __init__(self, code: DockerDiagnosticCodeV1):
        if type(code) is not DockerDiagnosticCodeV1:
            raise TypeError("exact Docker diagnostic code required")
        self.code = code
        super().__init__(code.value)


def closed_error(code: DockerDiagnosticCodeV1) -> DockerProviderError:
    return DockerProviderError(code)


def _exact_positive(value: object, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} is outside its canonical bound")
    return value


def _exact_tuple(values: object, name: str, *, maximum: int) -> tuple[str, ...]:
    if type(values) is not tuple or len(values) > maximum or any(type(v) is not str for v in values):
        raise TypeError(f"{name} must be an exact bounded tuple")
    result = tuple(safe_ref(value, name) for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must be unique")
    return result


@dataclass(frozen=True, slots=True)
class DockerImageV1:
    image_ref: str
    image_digest: str
    presence_policy: str = "present_only"

    def __post_init__(self) -> None:
        safe_ref(self.image_ref, "image_ref")
        if self.image_ref == "latest" or self.image_ref.endswith(":latest") or not _IMAGE_DIGEST.fullmatch(self.image_digest):
            raise ValueError("image must be pinned by exact sha256 digest")
        if self.presence_policy != "present_only":
            raise ValueError("Docker v1 never pulls or builds images")

    def to_dict(self) -> dict[str, object]:
        return {"image_ref": self.image_ref, "image_digest": self.image_digest,
                "presence_policy": self.presence_policy}

    @property
    def digest(self) -> str:
        return domain_digest("synaptic-docker-image/v1", canonical_bytes(self.to_dict()))


@dataclass(frozen=True, slots=True)
class DockerRuntimeV1:
    cpu_count: int
    memory_bytes: int
    timeout_seconds: int
    network_mode: str = "none"
    gpu_enabled: bool = False

    def __post_init__(self) -> None:
        _exact_positive(self.cpu_count, "cpu_count", 256)
        _exact_positive(self.memory_bytes, "memory_bytes", 2**50)
        _exact_positive(self.timeout_seconds, "timeout_seconds", 7 * 24 * 3600)
        if self.network_mode != "none" or self.gpu_enabled is not False:
            raise ValueError("Docker v1 is CPU-only and network-disabled")

    def to_dict(self) -> dict[str, object]:
        return {"cpu_count": self.cpu_count, "memory_bytes": self.memory_bytes,
                "timeout_seconds": self.timeout_seconds, "network_mode": self.network_mode,
                "gpu_enabled": self.gpu_enabled}

    @property
    def digest(self) -> str:
        return domain_digest("synaptic-docker-runtime/v1", canonical_bytes(self.to_dict()))


@dataclass(frozen=True, slots=True)
class DockerWorkloadV1:
    arguments: tuple[str, ...]
    environment_keys: tuple[str, ...]
    workload_digest: str

    def __post_init__(self) -> None:
        if type(self.arguments) is not tuple or not self.arguments or len(self.arguments) > MAX_ARGUMENTS:
            raise ValueError("arguments must be a nonempty bounded exact tuple")
        if any(type(value) is not str or not value or "\x00" in value for value in self.arguments):
            raise ValueError("arguments contain an invalid value")
        if sum(len(value.encode("utf-8")) for value in self.arguments) > MAX_ARGUMENT_BYTES:
            raise ValueError("arguments exceed the byte bound")
        keys = _exact_tuple(self.environment_keys, "environment_keys", maximum=MAX_ENVIRONMENT_KEYS)
        if keys != tuple(sorted(keys)):
            raise ValueError("environment keys must be canonical ascending")
        digest_text(self.workload_digest, "workload_digest")

    def to_dict(self) -> dict[str, object]:
        return {"arguments": list(self.arguments), "environment_keys": list(self.environment_keys),
                "workload_digest": self.workload_digest}


@dataclass(frozen=True, slots=True)
class DockerRootsV1:
    source_ref: str
    artifact_ref: str
    source_read_only: bool = True

    def __post_init__(self) -> None:
        safe_ref(self.source_ref, "source_ref")
        safe_ref(self.artifact_ref, "artifact_ref")
        if self.source_read_only is not True or self.source_ref == self.artifact_ref:
            raise ValueError("source must be read-only and distinct from artifact root")

    def to_dict(self) -> dict[str, object]:
        return {"source_ref": self.source_ref, "artifact_ref": self.artifact_ref,
                "source_read_only": self.source_read_only}


@dataclass(frozen=True, slots=True)
class DockerArtifactContractV1:
    roles: tuple[str, ...]
    maximum_artifact_bytes: int = MAX_ARTIFACT_BYTES
    maximum_total_bytes: int = MAX_ARTIFACT_TOTAL_BYTES

    def __post_init__(self) -> None:
        roles = _exact_tuple(self.roles, "artifact_roles", maximum=MAX_ARTIFACTS)
        if not roles or roles != tuple(sorted(roles)):
            raise ValueError("artifact roles must be nonempty canonical ascending")
        _exact_positive(self.maximum_artifact_bytes, "maximum_artifact_bytes", MAX_ARTIFACT_BYTES)
        _exact_positive(self.maximum_total_bytes, "maximum_total_bytes", MAX_ARTIFACT_TOTAL_BYTES)
        if self.maximum_artifact_bytes > self.maximum_total_bytes:
            raise ValueError("artifact size bound exceeds total bound")

    def to_dict(self) -> dict[str, object]:
        return {"roles": list(self.roles), "maximum_artifact_bytes": self.maximum_artifact_bytes,
                "maximum_total_bytes": self.maximum_total_bytes}

    @property
    def digest(self) -> str:
        return domain_digest("synaptic-docker-artifact-contract/v1", canonical_bytes(self.to_dict()))


@dataclass(frozen=True, slots=True)
class DockerProfileV1:
    provider: ProviderRef
    descriptor: ProviderDescriptor
    scope: ExecutionScopeV1
    executor_descriptor: ExecutorDescriptorV1
    adapter_descriptor: AdapterDescriptorV1
    image: DockerImageV1
    runtime: DockerRuntimeV1
    workload: DockerWorkloadV1
    roots: DockerRootsV1
    artifacts: DockerArtifactContractV1
    resource_digest: str
    quote_digest: str
    secret_requirements_digest: str

    def __post_init__(self) -> None:
        exact = (type(self.provider) is ProviderRef, type(self.descriptor) is ProviderDescriptor,
                 type(self.descriptor.capabilities) is ProviderCapabilities,
                 type(self.scope) is ExecutionScopeV1, type(self.executor_descriptor) is ExecutorDescriptorV1,
                 type(self.adapter_descriptor) is AdapterDescriptorV1, type(self.image) is DockerImageV1,
                 type(self.runtime) is DockerRuntimeV1, type(self.workload) is DockerWorkloadV1,
                 type(self.roots) is DockerRootsV1, type(self.artifacts) is DockerArtifactContractV1)
        if not all(exact):
            raise TypeError("profile contains a noncanonical value")
        if (self.provider.provider_id, self.descriptor.provider_id,
            self.executor_descriptor.provider_id, self.adapter_descriptor.provider_id) != (self.provider.provider_id,) * 4:
            raise ValueError("profile provider identities differ")
        for name in ("resource_digest", "quote_digest", "secret_requirements_digest"):
            digest_text(getattr(self, name), name)

    @classmethod
    def build(cls, *, provider: ProviderRef, descriptor: ProviderDescriptor,
              scope: ExecutionScopeV1, executor_descriptor: ExecutorDescriptorV1,
              adapter_descriptor: AdapterDescriptorV1, image: DockerImageV1,
              runtime: DockerRuntimeV1, workload: DockerWorkloadV1,
              roots: DockerRootsV1, artifacts: DockerArtifactContractV1,
              resource_digest: str, quote_digest: str,
              secret_requirements_digest: str) -> "DockerProfileV1":
        return cls(
            provider, descriptor, scope, executor_descriptor, adapter_descriptor,
            image, runtime, workload, roots, artifacts, resource_digest,
            quote_digest, secret_requirements_digest,
        )

    def to_dict(self) -> dict[str, object]:
        return _profile_document(validated_profile_snapshot(self))

    @property
    def profile_digest(self) -> str:
        snapshot = validated_profile_snapshot(self)
        return domain_digest(
            "synaptic-docker-profile/v1", canonical_bytes(_profile_document(snapshot))
        )


def _profile_document(profile: DockerProfileV1) -> dict[str, object]:
    capabilities = profile.descriptor.capabilities
    return {
            "schema_version": "synaptic-docker-profile/v1",
            "provider": {
                "provider_id": profile.provider.provider_id,
                "profile_ref": profile.provider.profile_ref,
            },
            "descriptor": {
                "schema_version": profile.descriptor.schema_version,
                "provider_id": profile.descriptor.provider_id,
                "display_name": profile.descriptor.display_name,
                "implementation_version": profile.descriptor.implementation_version,
                "capabilities": {
                    "observe": capabilities.observe, "logs": capabilities.logs,
                    "cancel": capabilities.cancel, "reconcile": capabilities.reconcile,
                    "artifact_streaming": capabilities.artifact_streaming,
                    "cost_quote": capabilities.cost_quote,
                },
            },
            "scope": {
                "account_ref": profile.scope.account_ref,
                "namespace_ref": profile.scope.namespace_ref,
            },
            "executor_descriptor": {
                "provider_id": profile.executor_descriptor.provider_id,
                "executor_id": profile.executor_descriptor.executor_id,
                "implementation_version": profile.executor_descriptor.implementation_version,
            },
            "adapter_descriptor": {
                "provider_id": profile.adapter_descriptor.provider_id,
                "adapter_id": profile.adapter_descriptor.adapter_id,
                "implementation_version": profile.adapter_descriptor.implementation_version,
            },
            "image": {
                "image_ref": profile.image.image_ref,
                "image_digest": profile.image.image_digest,
                "presence_policy": profile.image.presence_policy,
            },
            "runtime": {
                "cpu_count": profile.runtime.cpu_count,
                "memory_bytes": profile.runtime.memory_bytes,
                "timeout_seconds": profile.runtime.timeout_seconds,
                "network_mode": profile.runtime.network_mode,
                "gpu_enabled": profile.runtime.gpu_enabled,
            },
            "workload": {
                "arguments": list(profile.workload.arguments),
                "environment_keys": list(profile.workload.environment_keys),
                "workload_digest": profile.workload.workload_digest,
            },
            "roots": {
                "source_ref": profile.roots.source_ref,
                "artifact_ref": profile.roots.artifact_ref,
                "source_read_only": profile.roots.source_read_only,
            },
            "artifacts": {
                "roles": list(profile.artifacts.roles),
                "maximum_artifact_bytes": profile.artifacts.maximum_artifact_bytes,
                "maximum_total_bytes": profile.artifacts.maximum_total_bytes,
            },
            "resource_digest": profile.resource_digest,
            "quote_digest": profile.quote_digest,
            "secret_requirements_digest": profile.secret_requirements_digest,
        }


def validated_profile_snapshot(profile: DockerProfileV1) -> DockerProfileV1:
    if type(profile) is not DockerProfileV1:
        raise TypeError("exact Docker profile required")
    if (type(profile.provider) is not ProviderRef
            or type(profile.descriptor) is not ProviderDescriptor
            or type(profile.descriptor.capabilities) is not ProviderCapabilities
            or type(profile.scope) is not ExecutionScopeV1
            or type(profile.executor_descriptor) is not ExecutorDescriptorV1
            or type(profile.adapter_descriptor) is not AdapterDescriptorV1
            or type(profile.image) is not DockerImageV1
            or type(profile.runtime) is not DockerRuntimeV1
            or type(profile.workload) is not DockerWorkloadV1
            or type(profile.roots) is not DockerRootsV1
            or type(profile.artifacts) is not DockerArtifactContractV1):
        raise TypeError("profile contains a noncanonical value")
    capabilities = ProviderCapabilities(
        profile.descriptor.capabilities.observe,
        profile.descriptor.capabilities.logs,
        profile.descriptor.capabilities.cancel,
        profile.descriptor.capabilities.reconcile,
        profile.descriptor.capabilities.artifact_streaming,
        profile.descriptor.capabilities.cost_quote,
    )
    return DockerProfileV1.build(
        provider=ProviderRef(profile.provider.provider_id, profile.provider.profile_ref),
        descriptor=ProviderDescriptor(
            profile.descriptor.schema_version, profile.descriptor.provider_id,
            profile.descriptor.display_name, profile.descriptor.implementation_version,
            capabilities,
        ),
        scope=ExecutionScopeV1(profile.scope.account_ref, profile.scope.namespace_ref),
        executor_descriptor=ExecutorDescriptorV1(
            profile.executor_descriptor.provider_id,
            profile.executor_descriptor.executor_id,
            profile.executor_descriptor.implementation_version,
        ),
        adapter_descriptor=AdapterDescriptorV1(
            profile.adapter_descriptor.provider_id,
            profile.adapter_descriptor.adapter_id,
            profile.adapter_descriptor.implementation_version,
        ),
        image=DockerImageV1(
            profile.image.image_ref, profile.image.image_digest,
            profile.image.presence_policy,
        ),
        runtime=DockerRuntimeV1(
            profile.runtime.cpu_count, profile.runtime.memory_bytes,
            profile.runtime.timeout_seconds, profile.runtime.network_mode,
            profile.runtime.gpu_enabled,
        ),
        workload=DockerWorkloadV1(
            profile.workload.arguments, profile.workload.environment_keys,
            profile.workload.workload_digest,
        ),
        roots=DockerRootsV1(
            profile.roots.source_ref, profile.roots.artifact_ref,
            profile.roots.source_read_only,
        ),
        artifacts=DockerArtifactContractV1(
            profile.artifacts.roles, profile.artifacts.maximum_artifact_bytes,
            profile.artifacts.maximum_total_bytes,
        ),
        resource_digest=profile.resource_digest,
        quote_digest=profile.quote_digest,
        secret_requirements_digest=profile.secret_requirements_digest,
    )


@dataclass(frozen=True, slots=True)
class PreparedDockerPlanV1:
    profile: DockerProfileV1
    project_ref: str
    run_id: str
    plan_fingerprint: str
    source_digest: str
    preparation_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile", validated_profile_snapshot(self.profile))
        safe_ref(self.project_ref, "project_ref")
        safe_ref(self.run_id, "run_id")
        digest_text(self.plan_fingerprint, "plan_fingerprint")
        digest_text(self.source_digest, "source_digest")
        digest_text(self.preparation_digest, "preparation_digest")

    def to_dict(self) -> dict[str, object]:
        return {"provider": self.profile.provider.to_dict(), "project_ref": self.project_ref,
                "run_id": self.run_id, "plan_fingerprint": self.plan_fingerprint,
                "source_digest": self.source_digest, "preparation_digest": self.preparation_digest,
                "image_digest": self.profile.image.digest,
                "runtime_digest": self.profile.runtime.digest,
                "workload_digest": self.profile.workload.workload_digest,
                "roots": self.profile.roots.to_dict(), "artifact_contract_digest": self.profile.artifacts.digest}

    @property
    def digest(self) -> str:
        return domain_digest("synaptic-docker-prepared-plan/v1", canonical_bytes(self.to_dict()))


@dataclass(frozen=True, slots=True)
class DockerEffectIdentityV1:
    command_digest: str
    effect_id: str
    effect_kind: str
    plan: PreparedDockerPlanV1

    def __post_init__(self) -> None:
        digest_text(self.command_digest, "command_digest")
        safe_ref(self.effect_id, "effect_id")
        if self.effect_kind not in {"stage", "submit", "cancel"} or type(self.plan) is not PreparedDockerPlanV1:
            raise ValueError("effect identity is invalid")

    def to_dict(self) -> dict[str, object]:
        p = self.plan.profile
        return {
            "command_digest": self.command_digest, "effect_id": self.effect_id,
            "effect_kind": self.effect_kind, "prepared_plan_digest": self.plan.digest,
            "preparation_digest": self.plan.preparation_digest,
            "provider_id": p.provider.provider_id, "profile_ref": p.provider.profile_ref,
            "account_ref": p.scope.account_ref, "namespace_ref": p.scope.namespace_ref,
            "project_ref": self.plan.project_ref, "run_id": self.plan.run_id,
            "plan_fingerprint": self.plan.plan_fingerprint,
            "executor_descriptor_digest": p.executor_descriptor.digest,
            "adapter_descriptor_digest": p.adapter_descriptor.digest,
        }

    @property
    def digest(self) -> str:
        return domain_digest("synaptic-docker-effect-identity/v1", canonical_bytes(self.to_dict()))


@dataclass(frozen=True, slots=True)
class DockerCommandBindingV1:
    identity: DockerEffectIdentityV1
    command_bytes: bytes
    original_submit_command_bytes: bytes | None = None
    cancel_container_ref: str | None = None
    cancel_reason_digest: str | None = None
    cancel_submit_labels: "DockerLabelsV1 | None" = None
    cancel_authorization_digest: str | None = None

    def __post_init__(self) -> None:
        if type(self.identity) is not DockerEffectIdentityV1:
            raise TypeError("exact effect identity required")
        if type(self.command_bytes) is not bytes or not self.command_bytes or len(self.command_bytes) > 262144:
            raise ValueError("canonical command bytes invalid")
        cancel = self.identity.effect_kind == "cancel"
        if cancel != (type(self.original_submit_command_bytes) is bytes):
            raise ValueError("original submit command bytes matrix invalid")
        if self.original_submit_command_bytes is not None and (
            not self.original_submit_command_bytes or len(self.original_submit_command_bytes) > 262144
        ):
            raise ValueError("original submit command bytes invalid")
        if cancel != (self.cancel_container_ref is not None and self.cancel_reason_digest is not None
                      and type(self.cancel_submit_labels) is DockerLabelsV1
                      and self.cancel_authorization_digest is not None):
            raise ValueError("cancel target matrix invalid")
        if self.cancel_container_ref is not None:
            safe_ref(self.cancel_container_ref, "cancel_container_ref")
        if self.cancel_reason_digest is not None:
            digest_text(self.cancel_reason_digest, "cancel_reason_digest")
        if self.cancel_authorization_digest is not None:
            digest_text(self.cancel_authorization_digest, "cancel_authorization_digest")

    @property
    def command_digest(self): return self.identity.command_digest
    @property
    def effect_id(self): return self.identity.effect_id
    @property
    def effect_kind(self): return self.identity.effect_kind
    @property
    def plan(self): return self.identity.plan

    @property
    def binding_digest(self) -> str:
        p = self.plan.profile
        return domain_digest("synaptic-docker-command-binding/v1", canonical_bytes({
            "identity": self.identity.to_dict(),
            "prepared_plan": self.plan.to_dict(),
            "profile_digest": p.profile_digest,
            "resource_digest": p.resource_digest,
            "quote_digest": p.quote_digest,
            "secret_requirements_digest": p.secret_requirements_digest,
            "command_bytes_digest": domain_digest(
                "synaptic-docker-command-bytes/v1", self.command_bytes,
            ),
            "original_submit_command_bytes_digest": (
                None if self.original_submit_command_bytes is None else domain_digest(
                    "synaptic-docker-original-submit-command-bytes/v1",
                    self.original_submit_command_bytes,
                )
            ),
            "cancel_container_ref": self.cancel_container_ref,
            "cancel_reason_digest": self.cancel_reason_digest,
            "cancel_submit_labels": (
                None if self.cancel_submit_labels is None
                else self.cancel_submit_labels.to_dict()
            ),
            "cancel_authorization_digest": self.cancel_authorization_digest,
        }))


@dataclass(frozen=True, slots=True)
class AuthenticatedDockerCommandBindingV1:
    content: DockerCommandBindingV1
    binding_digest: str
    authority_ref: str
    key_ref: str
    tag: str

    def __post_init__(self) -> None:
        if type(self.content) is not DockerCommandBindingV1:
            raise TypeError("exact Docker command binding required")
        digest_text(self.binding_digest, "binding_digest")
        if self.binding_digest != self.content.binding_digest:
            raise ValueError("binding digest mismatch")
        safe_ref(self.authority_ref, "authority_ref")
        safe_ref(self.key_ref, "key_ref")
        digest_text(self.tag, "tag")

    @property
    def proof_digest(self) -> str:
        return domain_digest("synaptic-authenticated-docker-command-binding/v1", canonical_bytes({
            "binding_digest": self.binding_digest,
            "authority_ref": self.authority_ref,
            "key_ref": self.key_ref,
            "tag": self.tag,
        }))


@dataclass(frozen=True, slots=True)
class DockerLabelsV1:
    command_digest: str
    provider_id: str
    profile_ref: str
    account_ref: str
    namespace_ref: str
    project_ref: str
    run_id: str
    plan_fingerprint: str
    preparation_digest: str
    effect_id: str
    effect_kind: str
    effect_identity_digest: str
    adapter_descriptor_digest: str

    def __post_init__(self) -> None:
        digest_text(self.command_digest, "command_digest")
        for name in ("command_digest", "plan_fingerprint", "preparation_digest",
                     "effect_identity_digest", "adapter_descriptor_digest"):
            digest_text(getattr(self, name), name)
        if self.effect_kind not in {"stage", "submit", "cancel"}:
            raise ValueError("effect_kind invalid")
        for name in ("provider_id", "profile_ref", "account_ref", "namespace_ref", "project_ref", "run_id", "effect_id"):
            safe_ref(getattr(self, name), name)

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def digest(self) -> str:
        return domain_digest("synaptic-docker-labels/v1", canonical_bytes(self.to_dict()))

    @property
    def container_name(self) -> str:
        return "synaptic-" + self.command_digest[:24]


class DockerCreateDispositionV1(str, Enum):
    CREATED = "created"
    COLLISION = "collision"
    INDETERMINATE = "indeterminate"


class DockerLookupDispositionV1(str, Enum):
    FOUND = "found"
    DEFINITELY_ABSENT = "definitely_absent"
    INDETERMINATE = "indeterminate"
    MULTIPLE = "multiple"


class DockerLookupPurposeV1(str, Enum):
    RECONCILE_STAGE = "reconcile_stage"
    RECONCILE_SUBMIT = "reconcile_submit"
    RECONCILE_CANCEL = "reconcile_cancel"
    OBSERVE = "observe"


class DockerRunPhaseV1(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DockerLogTerminalPhaseV1(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class DockerLookupRequestV1:
    labels: DockerLabelsV1
    purpose: DockerLookupPurposeV1
    generation: int

    def __post_init__(self) -> None:
        if type(self.labels) is not DockerLabelsV1 or type(self.purpose) is not DockerLookupPurposeV1:
            raise TypeError("lookup request types invalid")
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("lookup generation invalid")

    @property
    def digest(self) -> str:
        return domain_digest("synaptic-docker-lookup-request/v1", canonical_bytes({
            "labels_digest": self.labels.digest, "purpose": self.purpose.value,
            "generation": self.generation,
        }))


@dataclass(frozen=True, slots=True)
class DockerAbsenceContentV1:
    request_digest: str
    labels_digest: str
    purpose: DockerLookupPurposeV1
    generation: int
    evidence_digest: str

    def __post_init__(self) -> None:
        for name in ("request_digest", "labels_digest", "evidence_digest"):
            digest_text(getattr(self, name), name)
        if type(self.purpose) is not DockerLookupPurposeV1 or type(self.generation) is not int or self.generation < 1:
            raise ValueError("absence binding invalid")

    @property
    def content_digest(self) -> str:
        return domain_digest("synaptic-docker-absence-content/v1", canonical_bytes({
            "request_digest": self.request_digest, "labels_digest": self.labels_digest,
            "purpose": self.purpose.value, "generation": self.generation,
            "evidence_digest": self.evidence_digest,
        }))


@dataclass(frozen=True, slots=True)
class AuthenticatedDockerAbsenceV1:
    content: DockerAbsenceContentV1
    authority_ref: str
    key_ref: str
    tag: str

    def __post_init__(self) -> None:
        if type(self.content) is not DockerAbsenceContentV1:
            raise TypeError("exact absence content required")
        safe_ref(self.authority_ref, "authority_ref"); safe_ref(self.key_ref, "key_ref")
        digest_text(self.tag, "tag")

    @property
    def proof_digest(self) -> str:
        return domain_digest("synaptic-authenticated-docker-absence/v1", canonical_bytes({
            "content_digest": self.content.content_digest, "authority_ref": self.authority_ref,
            "key_ref": self.key_ref, "tag": self.tag,
        }))


@dataclass(frozen=True, slots=True)
class DockerCreateResultV1:
    disposition: DockerCreateDispositionV1
    labels: DockerLabelsV1 | None = None
    container_ref: str | None = None

    def __post_init__(self) -> None:
        if type(self.disposition) is not DockerCreateDispositionV1:
            raise TypeError("exact create disposition required")
        found = self.disposition is DockerCreateDispositionV1.CREATED
        if found != (type(self.labels) is DockerLabelsV1 and type(self.container_ref) is str):
            raise ValueError("create result matrix invalid")
        if self.container_ref is not None:
            safe_ref(self.container_ref, "container_ref")


@dataclass(frozen=True, slots=True)
class DockerLookupResultV1:
    disposition: DockerLookupDispositionV1
    labels: DockerLabelsV1 | None = None
    container_ref: str | None = None
    phase: DockerRunPhaseV1 | None = None
    absence: AuthenticatedDockerAbsenceV1 | None = None

    def __post_init__(self) -> None:
        if type(self.disposition) is not DockerLookupDispositionV1:
            raise TypeError("exact lookup disposition required")
        found = self.disposition is DockerLookupDispositionV1.FOUND
        if found != (type(self.labels) is DockerLabelsV1 and type(self.container_ref) is str and type(self.phase) is DockerRunPhaseV1):
            raise ValueError("lookup result matrix invalid")
        if self.container_ref is not None:
            safe_ref(self.container_ref, "container_ref")
        absent = self.disposition is DockerLookupDispositionV1.DEFINITELY_ABSENT
        if absent != (type(self.absence) is AuthenticatedDockerAbsenceV1):
            raise ValueError("absence envelope matrix invalid")


@dataclass(frozen=True, slots=True)
class DockerSourceSealRequestV1:
    identity: DockerEffectIdentityV1
    source_ref: str
    source_digest: str

    def __post_init__(self) -> None:
        if type(self.identity) is not DockerEffectIdentityV1 or self.identity.effect_kind != "stage":
            raise ValueError("source seal requires exact STAGE identity")
        safe_ref(self.source_ref, "source_ref"); digest_text(self.source_digest, "source_digest")

    @property
    def digest(self) -> str:
        return domain_digest("synaptic-docker-source-seal-request/v1", canonical_bytes({
            "effect_identity_digest": self.identity.digest, "source_ref": self.source_ref,
            "source_digest": self.source_digest,
        }))


@dataclass(frozen=True, slots=True)
class DockerSourceSealContentV1:
    request_digest: str
    effect_identity_digest: str
    source_ref: str
    source_digest: str
    read_only: bool
    stage_ref: str
    evidence_digest: str

    def __post_init__(self) -> None:
        safe_ref(self.source_ref, "source_ref")
        safe_ref(self.stage_ref, "stage_ref")
        for name in ("request_digest", "effect_identity_digest", "source_digest", "evidence_digest"):
            digest_text(getattr(self, name), name)
        if self.read_only is not True:
            raise ValueError("source seal must prove read-only access")

    @property
    def content_digest(self) -> str:
        return domain_digest("synaptic-docker-source-seal-content/v1", canonical_bytes({
            "request_digest": self.request_digest, "effect_identity_digest": self.effect_identity_digest,
            "source_ref": self.source_ref, "source_digest": self.source_digest,
            "read_only": self.read_only, "stage_ref": self.stage_ref,
            "evidence_digest": self.evidence_digest,
        }))


@dataclass(frozen=True, slots=True)
class AuthenticatedDockerSourceSealV1:
    content: DockerSourceSealContentV1
    authority_ref: str
    key_ref: str
    tag: str

    def __post_init__(self) -> None:
        if type(self.content) is not DockerSourceSealContentV1:
            raise TypeError("exact source seal content required")
        safe_ref(self.authority_ref, "authority_ref"); safe_ref(self.key_ref, "key_ref"); digest_text(self.tag, "tag")


@dataclass(frozen=True, slots=True)
class DockerSourceSealLookupRequestV1:
    source_request: DockerSourceSealRequestV1
    generation: int

    def __post_init__(self) -> None:
        if type(self.source_request) is not DockerSourceSealRequestV1:
            raise TypeError("exact source seal request required")
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("source lookup generation invalid")

    @property
    def digest(self) -> str:
        return domain_digest("synaptic-docker-source-seal-lookup/v1", canonical_bytes({
            "source_request_digest": self.source_request.digest,
            "effect_identity_digest": self.source_request.identity.digest,
            "generation": self.generation,
        }))


@dataclass(frozen=True, slots=True)
class DockerSourceSealLookupResultV1:
    disposition: DockerLookupDispositionV1
    seal: AuthenticatedDockerSourceSealV1 | None = None
    absence: AuthenticatedDockerAbsenceV1 | None = None

    def __post_init__(self) -> None:
        if type(self.disposition) is not DockerLookupDispositionV1:
            raise TypeError("exact lookup disposition required")
        if self.disposition is DockerLookupDispositionV1.FOUND:
            valid = type(self.seal) is AuthenticatedDockerSourceSealV1 and self.absence is None
        elif self.disposition is DockerLookupDispositionV1.DEFINITELY_ABSENT:
            valid = type(self.absence) is AuthenticatedDockerAbsenceV1 and self.seal is None
        else:
            valid = self.seal is None and self.absence is None
        if not valid:
            raise ValueError("source lookup result matrix invalid")


@dataclass(frozen=True, slots=True)
class DockerCancellationRequestV1:
    cancellation_identity: DockerEffectIdentityV1
    submit_labels: DockerLabelsV1
    container_ref: str
    reason_digest: str
    authorization_digest: str

    def __post_init__(self) -> None:
        if type(self.cancellation_identity) is not DockerEffectIdentityV1 or self.cancellation_identity.effect_kind != "cancel":
            raise ValueError("cancellation identity invalid")
        if type(self.submit_labels) is not DockerLabelsV1 or self.submit_labels.effect_kind != "submit":
            raise ValueError("original submit labels required")
        identity = self.cancellation_identity
        p = identity.plan.profile
        if (
            self.submit_labels.provider_id, self.submit_labels.profile_ref,
            self.submit_labels.account_ref, self.submit_labels.namespace_ref,
            self.submit_labels.project_ref, self.submit_labels.run_id,
            self.submit_labels.plan_fingerprint, self.submit_labels.preparation_digest,
            self.submit_labels.adapter_descriptor_digest,
        ) != (
            p.provider.provider_id, p.provider.profile_ref, p.scope.account_ref,
            p.scope.namespace_ref, identity.plan.project_ref, identity.plan.run_id,
            identity.plan.plan_fingerprint, identity.plan.preparation_digest,
            p.adapter_descriptor.digest,
        ):
            raise ValueError("submit labels do not bind the cancellation target")
        safe_ref(self.container_ref, "container_ref"); digest_text(self.reason_digest, "reason_digest")
        digest_text(self.authorization_digest, "authorization_digest")

    @property
    def digest(self) -> str:
        return domain_digest("synaptic-docker-cancellation-request/v1", canonical_bytes({
            "cancellation_identity_digest": self.cancellation_identity.digest,
            "submit_labels_digest": self.submit_labels.digest, "container_ref": self.container_ref,
            "reason_digest": self.reason_digest,
            "authorization_digest": self.authorization_digest,
        }))


@dataclass(frozen=True, slots=True)
class DockerCancellationContentV1:
    request_digest: str
    cancellation_identity_digest: str
    submit_labels_digest: str
    container_ref: str
    reason_digest: str
    authorization_digest: str
    evidence_digest: str

    def __post_init__(self) -> None:
        for name in ("request_digest", "cancellation_identity_digest", "submit_labels_digest",
                     "reason_digest", "authorization_digest", "evidence_digest"):
            digest_text(getattr(self, name), name)
        safe_ref(self.container_ref, "container_ref")


@dataclass(frozen=True, slots=True)
class AuthenticatedDockerCancellationEvidenceV1:
    content: DockerCancellationContentV1
    authority_ref: str
    key_ref: str
    tag: str

    def __post_init__(self) -> None:
        if type(self.content) is not DockerCancellationContentV1:
            raise TypeError("exact cancellation content required")
        safe_ref(self.authority_ref, "authority_ref"); safe_ref(self.key_ref, "key_ref"); digest_text(self.tag, "tag")


@dataclass(frozen=True, slots=True)
class DockerCancellationLookupRequestV1:
    cancellation_request: DockerCancellationRequestV1
    generation: int

    def __post_init__(self) -> None:
        if type(self.cancellation_request) is not DockerCancellationRequestV1:
            raise TypeError("exact cancellation request required")
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("cancellation lookup generation invalid")

    @property
    def digest(self) -> str:
        return domain_digest("synaptic-docker-cancellation-lookup/v1", canonical_bytes({
            "cancellation_request_digest": self.cancellation_request.digest,
            "cancellation_identity_digest": self.cancellation_request.cancellation_identity.digest,
            "authorization_digest": self.cancellation_request.authorization_digest,
            "generation": self.generation,
        }))


@dataclass(frozen=True, slots=True)
class DockerCancellationAbsenceContentV1:
    lookup_request_digest: str
    cancellation_request_digest: str
    cancellation_identity_digest: str
    authorization_digest: str
    submit_labels_digest: str
    container_ref: str
    reason_digest: str
    generation: int
    resource_phase: DockerRunPhaseV1
    evidence_digest: str

    def __post_init__(self) -> None:
        for name in ("lookup_request_digest", "cancellation_request_digest",
                     "cancellation_identity_digest", "authorization_digest",
                     "submit_labels_digest", "reason_digest", "evidence_digest"):
            digest_text(getattr(self, name), name)
        safe_ref(self.container_ref, "container_ref")
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("cancellation absence generation invalid")
        if self.resource_phase is not DockerRunPhaseV1.RUNNING:
            raise ValueError("cancellation absence requires retained running resource")

    @property
    def content_digest(self) -> str:
        return domain_digest("synaptic-docker-cancellation-absence-content/v1", canonical_bytes({
            name: (getattr(self, name).value if name == "resource_phase" else getattr(self, name))
            for name in self.__dataclass_fields__
        }))


@dataclass(frozen=True, slots=True)
class AuthenticatedDockerCancellationAbsenceV1:
    content: DockerCancellationAbsenceContentV1
    authority_ref: str
    key_ref: str
    tag: str

    def __post_init__(self) -> None:
        if type(self.content) is not DockerCancellationAbsenceContentV1:
            raise TypeError("exact cancellation absence required")
        safe_ref(self.authority_ref, "authority_ref"); safe_ref(self.key_ref, "key_ref"); digest_text(self.tag, "tag")

    @property
    def proof_digest(self) -> str:
        return domain_digest("synaptic-authenticated-docker-cancellation-absence/v1", canonical_bytes({
            "content_digest": self.content.content_digest, "authority_ref": self.authority_ref,
            "key_ref": self.key_ref, "tag": self.tag,
        }))


@dataclass(frozen=True, slots=True)
class DockerCancellationLookupResultV1:
    disposition: DockerLookupDispositionV1
    evidence: AuthenticatedDockerCancellationEvidenceV1 | None = None
    absence: AuthenticatedDockerCancellationAbsenceV1 | None = None

    def __post_init__(self) -> None:
        if type(self.disposition) is not DockerLookupDispositionV1:
            raise TypeError("exact cancellation lookup disposition required")
        if self.disposition is DockerLookupDispositionV1.FOUND:
            valid = type(self.evidence) is AuthenticatedDockerCancellationEvidenceV1 and self.absence is None
        elif self.disposition is DockerLookupDispositionV1.DEFINITELY_ABSENT:
            valid = type(self.absence) is AuthenticatedDockerCancellationAbsenceV1 and self.evidence is None
        else:
            valid = self.evidence is None and self.absence is None
        if not valid:
            raise ValueError("cancellation lookup result matrix invalid")


@dataclass(frozen=True, slots=True)
class DockerArtifactEntryV1:
    descriptor: VerifiedArtifact
    relative_path: str
    file_identity_digest: str

    def __post_init__(self) -> None:
        if type(self.descriptor) is not VerifiedArtifact:
            raise TypeError("exact VerifiedArtifact required")
        digest_text(self.file_identity_digest, "file_identity_digest")
        safe_ref(self.relative_path, "relative_path")
        if "\\" in self.relative_path or self.relative_path.startswith("/") or any(
            part in {"", ".", ".."} for part in self.relative_path.split("/")
        ):
            raise ValueError("artifact path is not canonical relative POSIX")


@dataclass(frozen=True, slots=True)
class DockerArtifactInventoryV1:
    labels: DockerLabelsV1
    request_digest: str
    generation: int
    profile_digest: str
    prepared_plan_digest: str
    artifact_contract_digest: str
    artifact_root_ref: str
    entries: tuple[DockerArtifactEntryV1, ...]
    evidence_digest: str

    def __post_init__(self) -> None:
        if type(self.labels) is not DockerLabelsV1 or type(self.entries) is not tuple:
            raise TypeError("inventory types invalid")
        digest_text(self.request_digest, "request_digest")
        for name in ("profile_digest", "prepared_plan_digest", "artifact_contract_digest"):
            digest_text(getattr(self, name), name)
        safe_ref(self.artifact_root_ref, "artifact_root_ref")
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("inventory generation invalid")
        if not self.entries or len(self.entries) > MAX_ARTIFACTS or any(type(v) is not DockerArtifactEntryV1 for v in self.entries):
            raise ValueError("inventory entry set invalid")
        roles = tuple(v.descriptor.role for v in self.entries)
        paths = tuple(v.relative_path for v in self.entries)
        if roles != tuple(sorted(roles)) or len(roles) != len(set(roles)) or len(paths) != len(set(paths)):
            raise ValueError("inventory roles/paths are noncanonical")
        if any(v.descriptor.size_bytes > MAX_ARTIFACT_BYTES for v in self.entries) or sum(v.descriptor.size_bytes for v in self.entries) > MAX_ARTIFACT_TOTAL_BYTES:
            raise ValueError("inventory exceeds bounds")
        digest_text(self.evidence_digest, "evidence_digest")

    @property
    def content_digest(self) -> str:
        return domain_digest("synaptic-docker-artifact-inventory/v1", canonical_bytes({
            "labels_digest": self.labels.digest, "request_digest": self.request_digest,
            "generation": self.generation, "profile_digest": self.profile_digest,
            "prepared_plan_digest": self.prepared_plan_digest,
            "artifact_contract_digest": self.artifact_contract_digest,
            "artifact_root_ref": self.artifact_root_ref,
            "entries": [{"descriptor": value.descriptor.to_dict(), "relative_path": value.relative_path,
                         "file_identity_digest": value.file_identity_digest}
                        for value in self.entries], "evidence_digest": self.evidence_digest,
        }))


@dataclass(frozen=True, slots=True)
class AuthenticatedDockerArtifactInventoryV1:
    content: DockerArtifactInventoryV1
    authority_ref: str
    key_ref: str
    tag: str

    def __post_init__(self) -> None:
        if type(self.content) is not DockerArtifactInventoryV1:
            raise TypeError("exact artifact inventory required")
        safe_ref(self.authority_ref, "authority_ref"); safe_ref(self.key_ref, "key_ref"); digest_text(self.tag, "tag")


@dataclass(frozen=True, slots=True)
class DockerArtifactInventoryRequestV1:
    labels: DockerLabelsV1
    provider_request_digest: str
    generation: int
    profile_digest: str
    prepared_plan_digest: str
    artifact_contract_digest: str
    artifact_root_ref: str

    def __post_init__(self) -> None:
        if type(self.labels) is not DockerLabelsV1:
            raise TypeError("exact original labels required")
        for name in ("provider_request_digest", "profile_digest", "prepared_plan_digest",
                     "artifact_contract_digest"):
            digest_text(getattr(self, name), name)
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("inventory request generation invalid")
        safe_ref(self.artifact_root_ref, "artifact_root_ref")

    @property
    def digest(self) -> str:
        return domain_digest("synaptic-docker-artifact-inventory-request/v1", canonical_bytes({
            "labels_digest": self.labels.digest,
            "provider_request_digest": self.provider_request_digest,
            "generation": self.generation, "profile_digest": self.profile_digest,
            "prepared_plan_digest": self.prepared_plan_digest,
            "artifact_contract_digest": self.artifact_contract_digest,
            "artifact_root_ref": self.artifact_root_ref,
        }))


@dataclass(frozen=True, slots=True)
class DockerLogReadRequestV1:
    labels: DockerLabelsV1
    provider_request_digest: str
    query_digest: str
    after_sequence: int | None
    limit: int
    maximum_bytes: int
    maximum_entry_bytes: int
    generation: int

    def __post_init__(self) -> None:
        if type(self.labels) is not DockerLabelsV1:
            raise TypeError("exact labels required")
        digest_text(self.provider_request_digest, "provider_request_digest"); digest_text(self.query_digest, "query_digest")
        if self.after_sequence is not None and (type(self.after_sequence) is not int or self.after_sequence < 0):
            raise ValueError("log cursor invalid")
        if type(self.limit) is not int or not 1 <= self.limit <= MAX_LOG_ENTRIES:
            raise ValueError("log limit invalid")
        if type(self.maximum_bytes) is not int or not 4096 <= self.maximum_bytes <= MAX_LOG_BYTES:
            raise ValueError("log byte bound invalid")
        if type(self.maximum_entry_bytes) is not int or not 1 <= self.maximum_entry_bytes <= 4096:
            raise ValueError("log entry byte bound invalid")
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("log generation invalid")

    @property
    def digest(self) -> str:
        return domain_digest("synaptic-docker-log-read-request/v1", canonical_bytes({
            "labels_digest": self.labels.digest, "provider_request_digest": self.provider_request_digest,
            "query_digest": self.query_digest, "after_sequence": self.after_sequence,
            "limit": self.limit, "maximum_bytes": self.maximum_bytes, "generation": self.generation,
            "maximum_entry_bytes": self.maximum_entry_bytes,
        }))


@dataclass(frozen=True, slots=True)
class DockerLogPageV1:
    request_digest: str
    labels_digest: str
    log_stream_digest: str
    query_digest: str
    generation: int
    after_sequence: int | None
    requested_limit: int
    requested_maximum_bytes: int
    maximum_entry_bytes: int
    entries: tuple[object, ...]
    first_sequence: int | None
    last_sequence: int | None
    complete: bool
    truncated: bool
    next_sequence: int | None
    high_watermark_sequence: int
    terminal_phase: DockerLogTerminalPhaseV1 | None
    terminal_generation: int | None
    evidence_digest: str

    def __post_init__(self) -> None:
        from synaptic_tuner.api.v1.runs_facade import RunLogEntry
        for name in ("request_digest", "labels_digest", "log_stream_digest",
                     "query_digest", "evidence_digest"):
            digest_text(getattr(self, name), name)
        if type(self.generation) is not int or self.generation < 1 or type(self.entries) is not tuple:
            raise ValueError("log page binding invalid")
        if any(type(value) is not RunLogEntry for value in self.entries):
            raise TypeError("log entries must be exact RunLogEntry")
        if len(self.entries) > MAX_LOG_ENTRIES:
            raise ValueError("log page count exceeds bound")
        if type(self.requested_limit) is not int or not 1 <= self.requested_limit <= MAX_LOG_ENTRIES:
            raise ValueError("echoed log limit invalid")
        if type(self.requested_maximum_bytes) is not int or not 4096 <= self.requested_maximum_bytes <= MAX_LOG_BYTES:
            raise ValueError("echoed log bytes invalid")
        if type(self.maximum_entry_bytes) is not int or not 1 <= self.maximum_entry_bytes <= 4096:
            raise ValueError("echoed entry bound invalid")
        if len(self.entries) > self.requested_limit or any(value.size_bytes > self.maximum_entry_bytes for value in self.entries):
            raise ValueError("log page exceeds echoed bounds")
        sequences = tuple(value.sequence for value in self.entries)
        if any(a >= b for a, b in zip(sequences, sequences[1:])):
            raise ValueError("log page order invalid")
        expected_first = 1 if self.after_sequence is None else self.after_sequence + 1
        if sequences and sequences != tuple(range(expected_first, expected_first + len(sequences))):
            raise ValueError("log sequence must be contiguous from cursor")
        expected_bounds = ((sequences[0], sequences[-1]) if sequences else (None, None))
        if (self.first_sequence, self.last_sequence) != expected_bounds:
            raise ValueError("log first/last mismatch")
        if sum(value.size_bytes for value in self.entries) > self.requested_maximum_bytes:
            raise ValueError("log page exceeds requested bytes")
        if type(self.complete) is not bool or type(self.truncated) is not bool:
            raise ValueError("log page bounds invalid")
        if type(self.high_watermark_sequence) is not int or self.high_watermark_sequence < 0:
            raise ValueError("log high watermark invalid")
        cursor_floor = 0 if self.after_sequence is None else self.after_sequence
        if self.high_watermark_sequence < cursor_floor:
            raise ValueError("log high watermark precedes cursor")
        if (self.terminal_phase is None) != (self.terminal_generation is None):
            raise ValueError("terminal phase/generation matrix invalid")
        if self.terminal_phase is not None and type(self.terminal_phase) is not DockerLogTerminalPhaseV1:
            raise TypeError("exact terminal phase required")
        if self.terminal_generation is not None and (
            type(self.terminal_generation) is not int or self.terminal_generation < 1
            or self.terminal_generation != self.generation
        ):
            raise ValueError("terminal generation invalid")
        returned_end = sequences[-1] if sequences else cursor_floor
        complete_expected = returned_end == self.high_watermark_sequence
        if self.complete is not complete_expected or self.truncated is complete_expected:
            raise ValueError("log completion does not bind high watermark")
        if self.truncated:
            if (self.complete or not sequences or self.next_sequence != sequences[-1]
                    or sequences[-1] >= self.high_watermark_sequence):
                raise ValueError("truncated log page must be nonempty and advancing")
        elif (not self.complete or self.next_sequence is not None
              or returned_end != self.high_watermark_sequence):
            raise ValueError("complete log page must be untruncated without next cursor")

    @property
    def content_digest(self) -> str:
        return domain_digest("synaptic-docker-log-page/v1", canonical_bytes({
            "request_digest": self.request_digest, "labels_digest": self.labels_digest,
            "log_stream_digest": self.log_stream_digest,
            "query_digest": self.query_digest, "generation": self.generation,
            "after_sequence": self.after_sequence, "requested_limit": self.requested_limit,
            "requested_maximum_bytes": self.requested_maximum_bytes,
            "maximum_entry_bytes": self.maximum_entry_bytes,
            "entries": [value.to_dict() for value in self.entries],
            "first_sequence": self.first_sequence, "last_sequence": self.last_sequence,
            "complete": self.complete, "truncated": self.truncated,
            "next_sequence": self.next_sequence,
            "high_watermark_sequence": self.high_watermark_sequence,
            "terminal_phase": None if self.terminal_phase is None else self.terminal_phase.value,
            "terminal_generation": self.terminal_generation,
            "evidence_digest": self.evidence_digest,
        }))


@dataclass(frozen=True, slots=True)
class AuthenticatedDockerLogPageV1:
    content: DockerLogPageV1
    authority_ref: str
    key_ref: str
    tag: str

    def __post_init__(self) -> None:
        if type(self.content) is not DockerLogPageV1:
            raise TypeError("exact Docker log page required")
        safe_ref(self.authority_ref, "authority_ref"); safe_ref(self.key_ref, "key_ref"); digest_text(self.tag, "tag")


@dataclass(frozen=True, slots=True)
class DockerArtifactReadRequestV1:
    labels: DockerLabelsV1
    inventory_digest: str
    role: str
    relative_path: str
    maximum_bytes: int
    expected_size: int
    expected_sha256: str
    file_identity_digest: str
    generation: int
    profile_digest: str
    prepared_plan_digest: str
    artifact_contract_digest: str
    artifact_root_ref: str

    def __post_init__(self) -> None:
        if type(self.labels) is not DockerLabelsV1:
            raise TypeError("exact labels required")
        for name in ("inventory_digest", "expected_sha256", "file_identity_digest",
                     "profile_digest", "prepared_plan_digest", "artifact_contract_digest"):
            digest_text(getattr(self, name), name)
        safe_ref(self.role, "role"); safe_ref(self.relative_path, "relative_path")
        safe_ref(self.artifact_root_ref, "artifact_root_ref")
        if type(self.maximum_bytes) is not int or type(self.expected_size) is not int or not 0 <= self.expected_size <= self.maximum_bytes <= MAX_ARTIFACT_BYTES:
            raise ValueError("artifact read bounds invalid")
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("artifact generation invalid")

    @property
    def digest(self) -> str:
        return domain_digest("synaptic-docker-artifact-read-request/v1", canonical_bytes({
            "labels_digest": self.labels.digest, "inventory_digest": self.inventory_digest,
            "role": self.role, "relative_path": self.relative_path,
            "maximum_bytes": self.maximum_bytes, "expected_size": self.expected_size,
            "expected_sha256": self.expected_sha256,
            "file_identity_digest": self.file_identity_digest,
            "generation": self.generation, "profile_digest": self.profile_digest,
            "prepared_plan_digest": self.prepared_plan_digest,
            "artifact_contract_digest": self.artifact_contract_digest,
            "artifact_root_ref": self.artifact_root_ref,
        }))


@dataclass(frozen=True, slots=True)
class DockerArtifactChunkV1:
    stream_digest: str
    sequence: int
    offset: int
    data: bytes

    def __post_init__(self) -> None:
        digest_text(self.stream_digest, "stream_digest")
        if type(self.sequence) is not int or self.sequence < 0 or type(self.offset) is not int or self.offset < 0:
            raise ValueError("chunk position invalid")
        if type(self.data) is not bytes or not self.data or len(self.data) > 1_048_576:
            raise ValueError("chunk bytes invalid")


@dataclass(frozen=True, slots=True)
class DockerArtifactEOFV1:
    stream_digest: str
    next_sequence: int
    total_bytes: int
    sha256: str
    file_identity_digest: str
    evidence_digest: str
    authority_ref: str
    key_ref: str
    tag: str

    def __post_init__(self) -> None:
        for name in ("stream_digest", "sha256", "file_identity_digest", "evidence_digest", "tag"):
            digest_text(getattr(self, name), name)
        if type(self.next_sequence) is not int or self.next_sequence < 0 or type(self.total_bytes) is not int or self.total_bytes < 0:
            raise ValueError("EOF counters invalid")
        safe_ref(self.authority_ref, "authority_ref"); safe_ref(self.key_ref, "key_ref")


def labels_for(identity: DockerEffectIdentityV1) -> DockerLabelsV1:
    p = identity.plan.profile
    return DockerLabelsV1(
        identity.command_digest, p.provider.provider_id, p.provider.profile_ref,
        p.scope.account_ref, p.scope.namespace_ref, identity.plan.project_ref,
        identity.plan.run_id, identity.plan.plan_fingerprint,
        identity.plan.preparation_digest, identity.effect_id, identity.effect_kind,
        identity.digest, p.adapter_descriptor.digest,
    )
