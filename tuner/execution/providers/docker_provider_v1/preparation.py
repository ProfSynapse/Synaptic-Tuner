"""B3 binding and preparation translation for Docker provider v1."""

from __future__ import annotations

from synaptic_tuner.api.v1.planning import ProviderPlanContextV1, TrainingPlan
from synaptic_tuner.api.v1.providers import ProviderRef
from synaptic_tuner.api.v1.results import TrainingRunRef

from ...coordinator_v1.model import ProviderExecutionBindingV1
from ...foundation_v2.commands import CanonicalProviderPayloadV1
from ...foundation_v2.identities import EffectKind
from ...foundation_v2.preparation import CanonicalPreparationV2
from .model import DockerDiagnosticCodeV1, DockerProfileV1, DockerProviderError, PreparedDockerPlanV1


class DockerBindingResolverV1:
    def __init__(self, profile: DockerProfileV1):
        if type(profile) is not DockerProfileV1:
            raise TypeError("exact Docker profile required")
        self._profile = profile

    def resolve(self, provider: ProviderRef, context: ProviderPlanContextV1) -> ProviderExecutionBindingV1:
        try:
            if (type(provider) is not ProviderRef or type(context) is not ProviderPlanContextV1
                    or provider != self._profile.provider or context.provider != provider
                    or context.descriptor_digest != self._profile.descriptor.descriptor_digest
                    or context.profile_digest != self._profile.profile_digest):
                raise DockerProviderError(DockerDiagnosticCodeV1.BINDING_MISMATCH)
            p = self._profile
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
        self._profile = profile
        self._prepared: dict[str, PreparedDockerPlanV1] = {}

    def prepare(self, plan: TrainingPlan, run: TrainingRunRef,
                binding: ProviderExecutionBindingV1) -> CanonicalPreparationV2:
        try:
            p = self._profile
            expected = ProviderExecutionBindingV1(
                p.provider, p.descriptor.descriptor_digest, p.profile_digest, p.scope,
                p.executor_descriptor, p.adapter_descriptor.digest, p.resource_digest,
                p.quote_digest, p.secret_requirements_digest,
            )
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
            )
            self._prepared[value.preparation_digest] = PreparedDockerPlanV1(
                p, run.project_ref, run.run_id, plan.plan_fingerprint,
                plan.basis.source_digest, value.preparation_digest,
            )
            return value
        except DockerProviderError:
            raise
        except Exception:
            raise DockerProviderError(DockerDiagnosticCodeV1.INVALID_PLAN) from None

    def payload(self, preparation: CanonicalPreparationV2,
                kind: EffectKind) -> CanonicalProviderPayloadV1:
        try:
            if (type(preparation) is not CanonicalPreparationV2 or type(kind) is not EffectKind
                    or preparation.preparation_digest not in self._prepared):
                raise DockerProviderError(DockerDiagnosticCodeV1.BINDING_MISMATCH)
            return CanonicalProviderPayloadV1.build(
                self._profile.provider.provider_id, f"{kind.value}-payload/v2",
                preparation.workload_digest,
            )
        except DockerProviderError:
            raise
        except Exception:
            raise DockerProviderError(DockerDiagnosticCodeV1.BINDING_MISMATCH) from None

    def prepared(self, preparation_digest: str) -> PreparedDockerPlanV1:
        try:
            return self._prepared[preparation_digest]
        except Exception:
            raise DockerProviderError(DockerDiagnosticCodeV1.BINDING_MISMATCH) from None


__all__: list[str] = []
