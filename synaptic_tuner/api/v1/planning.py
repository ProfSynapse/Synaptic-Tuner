"""Acyclic provider-neutral planning contracts and digest boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ._contract import contract_digest, digest_text, exact_fields, required_text
from .providers import ProviderRef


@dataclass(frozen=True, slots=True)
class ResolvedTrainingRequest:
    """Closed references to host-resolved, immutable training inputs."""

    schema_version: str
    request_id: str
    project_ref: str
    source_digest: str
    resolved_config_digest: str
    workload_digest: str
    runtime_digest: str
    artifact_policy_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != "synaptic-resolved-training-request/v1":
            raise ValueError("unsupported resolved training request schema version")
        object.__setattr__(self, "request_id", required_text(self.request_id, "request_id"))
        object.__setattr__(self, "project_ref", required_text(self.project_ref, "project_ref"))
        for name in (
            "source_digest",
            "resolved_config_digest",
            "workload_digest",
            "runtime_digest",
            "artifact_policy_digest",
        ):
            object.__setattr__(self, name, digest_text(getattr(self, name), name))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "project_ref": self.project_ref,
            "source_digest": self.source_digest,
            "resolved_config_digest": self.resolved_config_digest,
            "workload_digest": self.workload_digest,
            "runtime_digest": self.runtime_digest,
            "artifact_policy_digest": self.artifact_policy_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ResolvedTrainingRequest":
        fields = frozenset(
            {
                "schema_version",
                "request_id",
                "project_ref",
                "source_digest",
                "resolved_config_digest",
                "workload_digest",
                "runtime_digest",
                "artifact_policy_digest",
            }
        )
        exact_fields(value, fields, "resolved_training_request")
        return cls(**{name: value[name] for name in fields})  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class TrainingPlanBasisV1:
    schema_version: str
    request_id: str
    project_ref: str
    source_digest: str
    resolved_config_digest: str
    workload_digest: str
    runtime_digest: str
    artifact_policy_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != "synaptic-training-plan-basis/v1":
            raise ValueError("unsupported training plan basis schema version")
        validated = ResolvedTrainingRequest(
            schema_version="synaptic-resolved-training-request/v1",
            request_id=self.request_id,
            project_ref=self.project_ref,
            source_digest=self.source_digest,
            resolved_config_digest=self.resolved_config_digest,
            workload_digest=self.workload_digest,
            runtime_digest=self.runtime_digest,
            artifact_policy_digest=self.artifact_policy_digest,
        )
        for name in (
            "request_id",
            "project_ref",
            "source_digest",
            "resolved_config_digest",
            "workload_digest",
            "runtime_digest",
            "artifact_policy_digest",
        ):
            object.__setattr__(self, name, getattr(validated, name))

    @classmethod
    def from_resolved(cls, value: ResolvedTrainingRequest) -> "TrainingPlanBasisV1":
        if not isinstance(value, ResolvedTrainingRequest):
            raise TypeError("value must be ResolvedTrainingRequest")
        document = value.to_dict()
        document["schema_version"] = "synaptic-training-plan-basis/v1"
        return cls.from_dict(document)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "project_ref": self.project_ref,
            "source_digest": self.source_digest,
            "resolved_config_digest": self.resolved_config_digest,
            "workload_digest": self.workload_digest,
            "runtime_digest": self.runtime_digest,
            "artifact_policy_digest": self.artifact_policy_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TrainingPlanBasisV1":
        fields = frozenset(
            {
                "schema_version",
                "request_id",
                "project_ref",
                "source_digest",
                "resolved_config_digest",
                "workload_digest",
                "runtime_digest",
                "artifact_policy_digest",
            }
        )
        exact_fields(value, fields, "training_plan_basis")
        return cls(**{name: value[name] for name in fields})  # type: ignore[arg-type]

    @property
    def basis_digest(self) -> str:
        return contract_digest("synaptic-training-plan-basis/v1", self.to_dict())


@dataclass(frozen=True, slots=True)
class ProviderPlanContextV1:
    schema_version: str
    provider: ProviderRef
    basis_digest: str
    descriptor_digest: str
    profile_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != "synaptic-provider-plan-context/v1":
            raise ValueError("unsupported provider plan context schema version")
        if not isinstance(self.provider, ProviderRef):
            raise TypeError("provider must be ProviderRef")
        for name in ("basis_digest", "descriptor_digest", "profile_digest"):
            object.__setattr__(self, name, digest_text(getattr(self, name), name))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "provider": self.provider.to_dict(),
            "basis_digest": self.basis_digest,
            "descriptor_digest": self.descriptor_digest,
            "profile_digest": self.profile_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ProviderPlanContextV1":
        fields = frozenset(
            {"schema_version", "provider", "basis_digest", "descriptor_digest", "profile_digest"}
        )
        exact_fields(value, fields, "provider_plan_context")
        provider = value["provider"]
        if not isinstance(provider, Mapping):
            raise TypeError("provider must be an object")
        return cls(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            provider=ProviderRef.from_dict(provider),
            basis_digest=value["basis_digest"],  # type: ignore[arg-type]
            descriptor_digest=value["descriptor_digest"],  # type: ignore[arg-type]
            profile_digest=value["profile_digest"],  # type: ignore[arg-type]
        )

    @property
    def provider_context_digest(self) -> str:
        return contract_digest("synaptic-provider-plan-context/v1", self.to_dict())


@dataclass(frozen=True, slots=True)
class ProviderPlanRef:
    context_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "context_digest", digest_text(self.context_digest, "context_digest")
        )

    def to_dict(self) -> dict[str, object]:
        return {"context_digest": self.context_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ProviderPlanRef":
        exact_fields(value, frozenset({"context_digest"}), "provider_plan_ref")
        return cls(context_digest=value["context_digest"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    schema_version: str
    basis: TrainingPlanBasisV1
    provider_plan: ProviderPlanRef

    def __post_init__(self) -> None:
        if self.schema_version != "synaptic-training-plan/v2":
            raise ValueError("unsupported training plan schema version")
        if not isinstance(self.basis, TrainingPlanBasisV1):
            raise TypeError("basis must be TrainingPlanBasisV1")
        if not isinstance(self.provider_plan, ProviderPlanRef):
            raise TypeError("provider_plan must be ProviderPlanRef")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "basis": self.basis.to_dict(),
            "provider_plan": self.provider_plan.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TrainingPlan":
        exact_fields(
            value, frozenset({"schema_version", "basis", "provider_plan"}), "training_plan"
        )
        basis = value["basis"]
        provider_plan = value["provider_plan"]
        if not isinstance(basis, Mapping) or not isinstance(provider_plan, Mapping):
            raise TypeError("basis and provider_plan must be objects")
        return cls(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            basis=TrainingPlanBasisV1.from_dict(basis),
            provider_plan=ProviderPlanRef.from_dict(provider_plan),
        )

    @property
    def plan_fingerprint(self) -> str:
        return contract_digest("synaptic-training-plan/v2", self.to_dict())


__all__ = [
    "ProviderPlanContextV1",
    "ProviderPlanRef",
    "ResolvedTrainingRequest",
    "TrainingPlan",
    "TrainingPlanBasisV1",
]
