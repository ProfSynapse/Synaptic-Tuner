"""B3 binding and preparation translation for Docker provider v1."""

from __future__ import annotations

from synaptic_tuner.api.v1.planning import ProviderPlanContextV1, TrainingPlan
from synaptic_tuner.api.v1.providers import ProviderRef
from synaptic_tuner.api.v1.results import TrainingRunRef

from ...coordinator_v1.model import ProviderExecutionBindingV1
from ...foundation_v2.commands import CanonicalProviderPayloadV1
from ...foundation_v2.identities import EffectKind
from ...foundation_v2.preparation import CanonicalPreparationV2
from .model import (
    DockerDiagnosticCodeV1, DockerProfileV1, DockerProviderError,
    PreparedDockerPlanV1, validated_profile_snapshot,
)


class DockerBindingResolverV1:
    def __init__(self, profile: DockerProfileV1):
        if type(profile) is not DockerProfileV1:
            raise TypeError("exact Docker profile required")
        self._profile = validated_profile_snapshot(profile)

    def resolve(self, provider: ProviderRef, context: ProviderPlanContextV1) -> ProviderExecutionBindingV1:
        try:
            profile = validated_profile_snapshot(self._profile)
            if (type(provider) is not ProviderRef or type(context) is not ProviderPlanContextV1
                    or provider != profile.provider or context.provider != provider
                    or context.descriptor_digest != profile.descriptor.descriptor_digest
                    or context.profile_digest != profile.profile_digest):
                raise DockerProviderError(DockerDiagnosticCodeV1.BINDING_MISMATCH)
            p = profile
            return ProviderExecutionBindingV1(
                p.provider, p.descriptor.descriptor_digest, p.profile_digest, p.scope,
                p.executor_descriptor, p.adapter_descriptor.digest, p.resource_digest,
                p.quote_digest, p.secret_requirements_digest,
            )
        except DockerProviderError:
            raise
        except Exception:
            raise DockerProviderError(DockerDiagnosticCodeV1.BINDING_MISMATCH) from None


class DockerPreparationMaterializerV1:
    def __init__(self, profile: DockerProfileV1):
        if type(profile) is not DockerProfileV1:
            raise TypeError("exact Docker profile required")
        self._profile = validated_profile_snapshot(profile)

    def _snapshot_binding(self) -> tuple[DockerProfileV1, ProviderExecutionBindingV1]:
        p = validated_profile_snapshot(self._profile)
        binding = ProviderExecutionBindingV1(
            p.provider, p.descriptor.descriptor_digest, p.profile_digest, p.scope,
            p.executor_descriptor, p.adapter_descriptor.digest, p.resource_digest,
            p.quote_digest, p.secret_requirements_digest,
        )
        return p, binding

    def _reconstruct(self, preparation: CanonicalPreparationV2) -> PreparedDockerPlanV1:
        """Validate a sealed preparation and deterministically recover its Docker plan."""
        try:
            if type(preparation) is not CanonicalPreparationV2:
                raise DockerProviderError(DockerDiagnosticCodeV1.BINDING_MISMATCH)
            raw = preparation.canonical_bytes
            sealed = CanonicalPreparationV2.parse(raw)
            p, expected_binding = self._snapshot_binding()
            if (sealed.canonical_bytes != raw
                    or sealed.provider != p.provider
                    or sealed.scope != p.scope
                    or sealed.workload_digest != p.workload.workload_digest
                    or sealed.runtime_digest != p.runtime.digest
                    or sealed.resource_digest != p.resource_digest
                    or sealed.artifact_contract_digest != p.artifacts.digest
                    or sealed.quote_digest != p.quote_digest
                    or sealed.secret_requirements_digest != p.secret_requirements_digest
                    or sealed.execution_binding_digest != expected_binding.binding_digest):
                raise DockerProviderError(DockerDiagnosticCodeV1.BINDING_MISMATCH)
            return PreparedDockerPlanV1(
                p, sealed.project_ref, sealed.run_id, sealed.plan_fingerprint,
                sealed.source_digest, sealed.preparation_digest,
            )
        except DockerProviderError:
            raise
        except Exception:
            raise DockerProviderError(DockerDiagnosticCodeV1.BINDING_MISMATCH) from None

    def prepare(self, plan: TrainingPlan, run: TrainingRunRef,
                binding: ProviderExecutionBindingV1) -> CanonicalPreparationV2:
        try:
            p, expected = self._snapshot_binding()
            if (type(plan) is not TrainingPlan or type(run) is not TrainingRunRef
                    or type(binding) is not ProviderExecutionBindingV1 or binding != expected
                    or plan.provider_plan.context_digest == ""
                    or run.project_ref != plan.basis.project_ref
                    or plan.basis.workload_digest != p.workload.workload_digest
                    or plan.basis.runtime_digest != p.runtime.digest
                    or plan.basis.artifact_policy_digest != p.artifacts.digest):
                raise DockerProviderError(DockerDiagnosticCodeV1.INVALID_PLAN)
            value = CanonicalPreparationV2.build(
                provider=p.provider, scope=p.scope, project_ref=run.project_ref,
                run_id=run.run_id, plan_fingerprint=plan.plan_fingerprint,
                source_digest=plan.basis.source_digest,
                workload_digest=p.workload.workload_digest, runtime_digest=p.runtime.digest,
                resource_digest=p.resource_digest, artifact_contract_digest=p.artifacts.digest,
                quote_digest=p.quote_digest, secret_requirements_digest=p.secret_requirements_digest,
                execution_binding_digest=expected.binding_digest,
            )
            return value
        except DockerProviderError:
            raise
        except Exception:
            raise DockerProviderError(DockerDiagnosticCodeV1.INVALID_PLAN) from None

    def payload(self, preparation: CanonicalPreparationV2,
                kind: EffectKind) -> CanonicalProviderPayloadV1:
        try:
            if type(kind) is not EffectKind:
                raise DockerProviderError(DockerDiagnosticCodeV1.BINDING_MISMATCH)
            prepared = self._reconstruct(preparation)
            return CanonicalProviderPayloadV1.build(
                prepared.profile.provider.provider_id, f"{kind.value}-payload/v2",
                prepared.profile.workload.workload_digest,
            )
        except DockerProviderError:
            raise
        except Exception:
            raise DockerProviderError(DockerDiagnosticCodeV1.BINDING_MISMATCH) from None

    def prepared(self, preparation: CanonicalPreparationV2) -> PreparedDockerPlanV1:
        return self._reconstruct(preparation)


__all__: list[str] = []
