"""B3 binding and preparation translation for Docker provider v1."""

from __future__ import annotations

import hashlib
from pathlib import PurePosixPath

from synaptic_tuner.api.v1.planning import (
    ProviderPlanContextV1,
    TrainingPlan as LegacyTrainingPlan,
)
from synaptic_tuner.api.v1.providers import ProviderRef
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.training import TrainingPlan

from ...coordinator_v1.model import ProviderExecutionBindingV1
from ...foundation_v2.canonical import canonical_bytes, digest_text
from ...foundation_v2.commands import CanonicalProviderPayloadV1
from ...foundation_v2.identities import EffectKind
from ...foundation_v2.preparation import CanonicalPreparationV2
from ....runtime.dispatch import (
    CanonicalWorkloadFileLocationV1,
    WorkerBundleMaterializationV1,
    WorkerControlLocationV1,
    build_source_worker_invocation,
    materialize_worker_bundle,
)
from .model import (
    DockerDiagnosticCodeV1, DockerProfileV1, DockerProviderError,
    PreparedDockerPlanV1, validated_profile_snapshot,
)


_DOCKER_CONTROL_ROOT = PurePosixPath("/source/control")
_ARTIFACT_ROLES = (
    "final_model",
    "tokenizer",
    "training_lineage",
    "training_metrics",
    "workload_record",
)


def _document_digest(value: dict[str, object]) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


_ZERO_QUOTE_DIGEST = _document_digest(
    {
        "schema_version": "synaptic-docker-local-quote/v1",
        "currency": "USD",
        "amount": "0",
    }
)
_NO_SECRET_REQUIREMENTS_DIGEST = _document_digest(
    {
        "schema_version": "synaptic-docker-secret-requirements/v1",
        "secrets": [],
    }
)


def _resource_digest(plan: TrainingPlan) -> str:
    return _document_digest(
        {
            "accelerator": plan.resources.accelerator,
            "accelerator_count": plan.resources.accelerator_count,
            "timeout_seconds": plan.resources.timeout_seconds,
        }
    )


def _profile_image(profile: DockerProfileV1) -> str:
    return f"{profile.image.image_ref}@{profile.image.image_digest}"


def _artifact_roles(plan: TrainingPlan) -> tuple[str, ...]:
    workload = plan.workload.to_dict()
    artifacts = workload.get("artifacts")
    requirements = artifacts.get("requirements") if type(artifacts) is dict else None
    if type(requirements) is not list or any(
        type(item) is not dict for item in requirements
    ):
        raise ValueError(
            "public training workload has no canonical artifact requirements"
        )
    roles = tuple(item.get("role") for item in requirements)
    if any(type(role) is not str for role in roles):
        raise ValueError("public training workload artifact roles are malformed")
    return tuple(sorted(roles))


def _execution_binding(profile: DockerProfileV1) -> ProviderExecutionBindingV1:
    return ProviderExecutionBindingV1(
        profile.provider,
        profile.descriptor.descriptor_digest,
        profile.profile_digest,
        profile.scope,
        profile.executor_descriptor,
        profile.adapter_descriptor.digest,
        profile.resource_digest,
        profile.quote_digest,
        profile.secret_requirements_digest,
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

    def prepare(self, plan: LegacyTrainingPlan, run: TrainingRunRef,
                binding: ProviderExecutionBindingV1) -> CanonicalPreparationV2:
        try:
            p, expected = self._snapshot_binding()
            if (type(plan) is not LegacyTrainingPlan or type(run) is not TrainingRunRef
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


class DockerTrainingPreparationBridgeV1:
    """Bind the public training plan directly to Docker provider v1."""

    def __init__(self, profile: DockerProfileV1):
        if type(profile) is not DockerProfileV1:
            raise TypeError("exact Docker profile required")
        self._profile = validated_profile_snapshot(profile)

    def _snapshot(self) -> DockerProfileV1:
        return validated_profile_snapshot(self._profile)

    def _expected(
        self,
        plan: TrainingPlan,
        run: TrainingRunRef,
        source_digest: str,
    ) -> tuple[
        DockerProfileV1,
        ProviderExecutionBindingV1,
        WorkerBundleMaterializationV1,
    ]:
        if type(plan) is not TrainingPlan:
            raise TypeError("exact public TrainingPlan required")
        if type(run) is not TrainingRunRef:
            raise TypeError("exact TrainingRunRef required")
        digest_text(source_digest, "source_digest")
        profile = self._snapshot()
        source = plan.execution_source
        control_location = WorkerControlLocationV1(_DOCKER_CONTROL_ROOT)
        worker = build_source_worker_invocation(
            plan,
            control_location,
            CanonicalWorkloadFileLocationV1(control_location.control_root),
        )
        bundle = materialize_worker_bundle(worker)
        environment_keys = tuple(sorted(dict(bundle.dispatch.environment)))
        image = _profile_image(profile)
        accelerator = profile.runtime.accelerator_devices
        if (
            profile.provider.provider_id != "docker"
            or profile.descriptor.capabilities.cost_quote is not False
            or run.run_id != source.run_id
            or profile.workload.workload_digest != bundle.workload_sha256
            or profile.workload.arguments != bundle.dispatch.argv
            or profile.workload.environment_keys != environment_keys
            or profile.roots.source_read_only is not True
            or image != plan.runtime.image
            or plan.runtime.python_version != source.python_version
            or profile.runtime.timeout_seconds != plan.resources.timeout_seconds
            or plan.resources.accelerator_count != 1
            or accelerator.kind != plan.resources.accelerator
            or profile.runtime.network_mode != "none"
            or profile.resource_digest != _resource_digest(plan)
            or _artifact_roles(plan) != _ARTIFACT_ROLES
            or tuple(sorted(plan.artifact_policy.required_kinds)) != _ARTIFACT_ROLES
            or profile.artifacts.roles != _ARTIFACT_ROLES
            or profile.quote_digest != _ZERO_QUOTE_DIGEST
            or source.secret_requirements_digest != _NO_SECRET_REQUIREMENTS_DIGEST
            or profile.secret_requirements_digest != _NO_SECRET_REQUIREMENTS_DIGEST
        ):
            raise DockerProviderError(DockerDiagnosticCodeV1.INVALID_PLAN)
        return profile, _execution_binding(profile), bundle

    def prepare(
        self,
        plan: TrainingPlan,
        run: TrainingRunRef,
        source_digest: str,
    ) -> CanonicalPreparationV2:
        try:
            profile, binding, bundle = self._expected(plan, run, source_digest)
            return CanonicalPreparationV2.build(
                provider=profile.provider,
                scope=profile.scope,
                project_ref=run.project_ref,
                run_id=run.run_id,
                plan_fingerprint=plan.fingerprint,
                source_digest=source_digest,
                workload_digest=bundle.workload_sha256,
                runtime_digest=profile.runtime.digest,
                resource_digest=profile.resource_digest,
                artifact_contract_digest=profile.artifacts.digest,
                quote_digest=profile.quote_digest,
                secret_requirements_digest=profile.secret_requirements_digest,
                execution_binding_digest=binding.binding_digest,
            )
        except DockerProviderError:
            raise
        except Exception:
            raise DockerProviderError(DockerDiagnosticCodeV1.INVALID_PLAN) from None

    def prepared(
        self,
        *,
        preparation: CanonicalPreparationV2,
        plan: TrainingPlan,
        run: TrainingRunRef,
        source_digest: str,
    ) -> PreparedDockerPlanV1:
        try:
            if type(preparation) is not CanonicalPreparationV2:
                raise DockerProviderError(DockerDiagnosticCodeV1.BINDING_MISMATCH)
            sealed = CanonicalPreparationV2.parse(preparation.canonical_bytes)
            expected = self.prepare(plan, run, source_digest)
            if (
                sealed.canonical_bytes != preparation.canonical_bytes
                or sealed.canonical_bytes != expected.canonical_bytes
                or sealed.preparation_digest != expected.preparation_digest
            ):
                raise DockerProviderError(DockerDiagnosticCodeV1.BINDING_MISMATCH)
            profile = self._snapshot()
            return PreparedDockerPlanV1(
                profile,
                run.project_ref,
                run.run_id,
                plan.fingerprint,
                source_digest,
                sealed.preparation_digest,
            )
        except Exception:
            raise DockerProviderError(DockerDiagnosticCodeV1.BINDING_MISMATCH) from None


__all__: list[str] = []
