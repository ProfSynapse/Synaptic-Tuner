"""One-shot Docker effects and lookup-only reconciliation."""

from __future__ import annotations

from threading import Lock
from dataclasses import fields

from ...foundation_v2.commands import CanonicalProviderPayloadV1, parse_exact_command
from ...foundation_v2.executors import (
    ExecutionResolutionRequestV2,
    ReconciliationResolutionRequestV2,
    mint_resolved_adapter,
    mint_resolved_executor,
)
from ...foundation_v2.identities import EffectKind
from ...foundation_v2.observations import ObservationDisposition, ProviderObservationV1
from ...foundation_v2.preparation import CanonicalPreparationV2
from ...foundation_v2.references import (
    CancellationRefV1,
    ProviderRunRefV1,
    ProviderStageRefV1,
    ScopedProviderRunRefV1,
)
from .model import (
    AuthenticatedDockerAbsenceV1,
    AuthenticatedDockerCancellationAbsenceV1,
    AuthenticatedDockerCancellationEvidenceV1,
    AuthenticatedDockerCommandBindingV1,
    AuthenticatedDockerSourceSealV1,
    DockerCancellationRequestV1,
    DockerCancellationContentV1,
    DockerCancellationLookupRequestV1,
    DockerCancellationLookupResultV1,
    DockerCancellationAbsenceContentV1,
    DockerAbsenceContentV1,
    DockerCommandBindingV1,
    DockerCreateDispositionV1,
    DockerCreateResultV1,
    DockerDiagnosticCodeV1,
    DockerEffectIdentityV1,
    DockerLookupDispositionV1,
    DockerLookupPurposeV1,
    DockerLookupRequestV1,
    DockerLookupResultV1,
    DockerProviderError,
    DockerRunPhaseV1,
    DockerSourceSealRequestV1,
    DockerSourceSealContentV1,
    DockerSourceSealLookupRequestV1,
    DockerSourceSealLookupResultV1,
    labels_for,
)


def _rebuilt(value, expected_type):
    if type(value) is not expected_type:
        raise ValueError
    rebuilt = expected_type(**{field.name: getattr(value, field.name) for field in fields(expected_type)})
    if rebuilt != value:
        raise ValueError
    return rebuilt


def _resolve_authenticated_binding(catalog, binding_authority, profile, command_digest,
                                   *, expected_command_bytes=None):
    owned = catalog.resolve(command_digest)
    if type(owned) is not AuthenticatedDockerCommandBindingV1:
        raise ValueError
    rebuilt_owned = AuthenticatedDockerCommandBindingV1(
        owned.content, owned.binding_digest, owned.authority_ref, owned.key_ref, owned.tag,
    )
    if (rebuilt_owned != owned or owned.binding_digest != owned.content.binding_digest
            or owned.authority_ref != binding_authority.authority_ref
            or owned.key_ref != binding_authority.key_ref
            or binding_authority.authenticate(owned) is not True):
        raise ValueError
    binding = _rebuilt(owned.content, DockerCommandBindingV1)
    command = parse_exact_command(binding.command_bytes)
    p = binding.plan.profile
    prep = command.preparation
    if (expected_command_bytes is not None and binding.command_bytes != expected_command_bytes):
        raise ValueError
    if (binding.command_digest, binding.effect_id, binding.effect_kind,
        binding.plan.profile, command.digest, command.operation.effect.effect_id,
        command.operation.effect.kind.value, prep.preparation_digest,
        prep.provider, prep.scope, prep.project_ref, prep.run_id,
        prep.plan_fingerprint, prep.source_digest, prep.workload_digest,
        prep.runtime_digest, prep.resource_digest, prep.artifact_contract_digest,
        prep.quote_digest, prep.secret_requirements_digest) != (
            command_digest, binding.identity.effect_id, binding.identity.effect_kind,
            profile, binding.identity.command_digest, binding.identity.effect_id,
            binding.identity.effect_kind, binding.plan.preparation_digest,
            p.provider, p.scope, binding.plan.project_ref, binding.plan.run_id,
            binding.plan.plan_fingerprint, binding.plan.source_digest,
            p.workload.workload_digest, p.runtime.digest, p.resource_digest,
            p.artifacts.digest, p.quote_digest, p.secret_requirements_digest,
    ):
        raise ValueError
    if binding.effect_kind == "cancel":
        submit = parse_exact_command(binding.original_submit_command_bytes)
        if submit.operation.effect.kind.value != "submit":
            raise ValueError
        submit_identity = DockerEffectIdentityV1(
            submit.digest, submit.operation.effect.effect_id, "submit", binding.plan,
        )
        cancellation = command.to_dict()["cancellation"]
        if (submit.preparation.preparation_digest != binding.plan.preparation_digest
                or labels_for(submit_identity) != binding.cancel_submit_labels
                or cancellation["provider_job_ref"] != binding.cancel_container_ref
                or cancellation["reason_digest"] != binding.cancel_reason_digest):
            raise ValueError
    elif binding.original_submit_command_bytes is not None:
        raise ValueError
    return binding, command


def _indeterminate(binding: DockerCommandBindingV1, request, epoch: int = 1) -> ProviderObservationV1:
    return ProviderObservationV1(
        binding.effect_id, binding.command_digest,
        binding.plan.profile.executor_descriptor.digest,
        ObservationDisposition.INDETERMINATE, request.digest, epoch,
    )


class DockerEffectExecutorV1:
    effect_kinds = ("stage", "submit", "cancel")
    payload_schemas = ("stage-payload/v2", "submit-payload/v2", "cancel-payload/v2")

    def __init__(self, profile, command_catalog, binding_authority, image_inventory, source, control,
                 cancellations, evidence_authority):
        self.descriptor = profile.executor_descriptor
        self.provider_id = profile.provider.provider_id
        self.profile_ref = profile.provider.profile_ref
        self.account_ref = profile.scope.account_ref
        self.namespace_ref = profile.scope.namespace_ref
        self._profile, self._catalog = profile, command_catalog
        self._binding_authority = binding_authority
        self._images, self._source, self._control = image_inventory, source, control
        self._cancellations = cancellations
        self._authority = evidence_authority
        self._lock = Lock()
        self._attempted: set[str] = set()

    def _binding(self, payload, request) -> DockerCommandBindingV1:
        try:
            if type(payload) is not CanonicalProviderPayloadV1 or type(request) is not ExecutionResolutionRequestV2:
                raise ValueError
            binding, _ = _resolve_authenticated_binding(
                self._catalog, self._binding_authority, self._profile, request.command_digest,
            )
            p = binding.plan.profile
            expected = (binding.command_digest, binding.effect_kind, p.provider.provider_id,
                        p.provider.profile_ref, p.scope.account_ref, p.scope.namespace_ref,
                        p.executor_descriptor.digest, p.workload.workload_digest,
                        f"{binding.effect_kind}-payload/v2")
            actual = (request.command_digest, request.effect_kind, request.provider_id,
                      request.profile_ref, request.account_ref, request.namespace_ref,
                      request.descriptor_digest, payload.input_digest, payload.payload_kind)
            if type(binding) is not DockerCommandBindingV1 or actual != expected or payload.provider_id != p.provider.provider_id:
                raise ValueError
            return binding
        except Exception:
            raise DockerProviderError(DockerDiagnosticCodeV1.BINDING_MISMATCH) from None

    def execute_once(self, payload: object, request: ExecutionResolutionRequestV2) -> ProviderObservationV1:
        binding = self._binding(payload, request)
        with self._lock:
            if binding.command_digest in self._attempted:
                return _indeterminate(binding, request)
            self._attempted.add(binding.command_digest)
        if binding.effect_kind == "stage":
            return self._stage(binding, request)
        if binding.effect_kind == "submit":
            return self._submit(binding, request)
        return self._cancel(binding, request)

    def _stage(self, binding, request):
        plan = binding.plan
        try:
            if self._images.require_present(plan.profile.image) is not True:
                raise DockerProviderError(DockerDiagnosticCodeV1.IMAGE_UNAVAILABLE)
        except DockerProviderError:
            raise
        except Exception:
            raise DockerProviderError(DockerDiagnosticCodeV1.IMAGE_UNAVAILABLE) from None
        seal_request = DockerSourceSealRequestV1(
            binding.identity, plan.profile.roots.source_ref, plan.source_digest
        )
        try:
            seal = self._source.seal_read_only(seal_request)
        except Exception:
            return _indeterminate(binding, request)
        try:
            content = (_rebuilt(seal.content, DockerSourceSealContentV1)
                       if type(seal) is AuthenticatedDockerSourceSealV1 else None)
            if (self._authority.authenticate_source_seal(seal) is not True
                    or (content.request_digest, content.effect_identity_digest,
                        content.source_ref, content.source_digest, content.read_only) != (
                        seal_request.digest, binding.identity.digest,
                        plan.profile.roots.source_ref, plan.source_digest, True)):
                raise DockerProviderError(DockerDiagnosticCodeV1.SOURCE_UNSEALED)
            return ProviderObservationV1(
                binding.effect_id, binding.command_digest, self.descriptor.digest,
                ObservationDisposition.FOUND, request.digest, 1,
                stage_ref=ProviderStageRefV1(
                    self.provider_id, self.profile_ref, self.account_ref,
                    self.namespace_ref, content.stage_ref,
                ),
            )
        except DockerProviderError:
            raise
        except Exception:
            raise DockerProviderError(DockerDiagnosticCodeV1.SOURCE_UNSEALED) from None

    def _submit(self, binding, request):
        plan = binding.plan
        labels = labels_for(binding.identity)
        try:
            result = self._control.create_once(
                labels=labels, image=plan.profile.image, runtime=plan.profile.runtime,
                workload=plan.profile.workload, source_ref=plan.profile.roots.source_ref,
                artifact_ref=plan.profile.roots.artifact_ref,
            )
        except Exception:
            return _indeterminate(binding, request)
        if type(result) is not DockerCreateResultV1:
            return _indeterminate(binding, request)
        if result.disposition is DockerCreateDispositionV1.COLLISION:
            raise DockerProviderError(DockerDiagnosticCodeV1.CREATE_COLLISION)
        if result.disposition is not DockerCreateDispositionV1.CREATED or result.labels != labels:
            return _indeterminate(binding, request)
        try:
            started = self._control.start_once(result.container_ref, labels)
        except Exception:
            started = False
        if started is not True:
            return _indeterminate(binding, request)
        return ProviderObservationV1(
            binding.effect_id, binding.command_digest, self.descriptor.digest,
            ObservationDisposition.FOUND, request.digest, 1,
            provider_run=ScopedProviderRunRefV1(
                self.provider_id, self.profile_ref, self.account_ref,
                self.namespace_ref, result.container_ref,
            ),
        )

    def _cancel(self, binding, request):
        cancel_request = DockerCancellationRequestV1(
                binding.identity, binding.cancel_submit_labels,
            binding.cancel_container_ref, binding.cancel_reason_digest,
            binding.cancel_authorization_digest,
        )
        try:
            evidence = self._cancellations.stop_once(cancel_request)
        except Exception:
            evidence = None
        try:
            content = (_rebuilt(evidence.content, DockerCancellationContentV1)
                       if type(evidence) is AuthenticatedDockerCancellationEvidenceV1 else None)
        except Exception:
            return _indeterminate(binding, request)
        expected = (cancel_request.digest, binding.identity.digest,
                    binding.cancel_submit_labels.digest, binding.cancel_container_ref,
                    binding.cancel_reason_digest, binding.cancel_authorization_digest)
        actual = None if content is None else (
            content.request_digest, content.cancellation_identity_digest,
            content.submit_labels_digest, content.container_ref, content.reason_digest,
            content.authorization_digest,
        )
        try:
            authenticated = self._authority.authenticate_cancellation(evidence) is True
        except Exception:
            authenticated = False
        if not authenticated or actual != expected:
            return _indeterminate(binding, request)
        return ProviderObservationV1(
            binding.effect_id, binding.command_digest, self.descriptor.digest,
            ObservationDisposition.FOUND, request.digest, 1,
            cancellation=CancellationRefV1(
                ProviderRunRefV1(binding.cancel_container_ref), binding.cancel_reason_digest
            ),
        )


class DockerExecutorResolverV1:
    def __init__(self, executor: DockerEffectExecutorV1):
        self._executor = executor

    def resolve(self, request: ExecutionResolutionRequestV2):
        return mint_resolved_executor(request, self._executor)


class DockerReconciliationAdapterV1:
    capabilities = ("lookup",)

    def __init__(self, profile, command_catalog, binding_authority, control, source_seals,
                 cancellations, evidence_authority):
        self.descriptor = profile.adapter_descriptor
        self.provider_id = profile.provider.provider_id
        self.profile_ref = profile.provider.profile_ref
        self.account_ref = profile.scope.account_ref
        self.namespace_ref = profile.scope.namespace_ref
        self._profile, self._catalog, self._control = profile, command_catalog, control
        self._binding_authority = binding_authority
        self._source_seals, self._cancellations = source_seals, cancellations
        self._authority = evidence_authority

    def lookup(self, target, preparation: CanonicalPreparationV2) -> ProviderObservationV1:
        try:
            binding, command = _resolve_authenticated_binding(
                self._catalog, self._binding_authority, self._profile,
                target.command_digest, expected_command_bytes=target.command_bytes,
            )
            p = binding.plan.profile
            if (command.digest, command.operation.effect.effect_id,
                preparation.preparation_digest, preparation.provider,
                preparation.project_ref, preparation.run_id,
                preparation.plan_fingerprint, preparation.source_digest,
                preparation.workload_digest, preparation.runtime_digest,
                preparation.resource_digest, preparation.artifact_contract_digest,
                preparation.quote_digest, preparation.secret_requirements_digest) != (
                    binding.command_digest, binding.effect_id,
                    binding.plan.preparation_digest, p.provider,
                    binding.plan.project_ref, binding.plan.run_id,
                    binding.plan.plan_fingerprint, binding.plan.source_digest,
                    p.workload.workload_digest, p.runtime.digest, p.resource_digest,
                    p.artifacts.digest, p.quote_digest, p.secret_requirements_digest):
                raise ValueError
            labels = (binding.cancel_submit_labels if binding.effect_kind == "cancel"
                      else labels_for(binding.identity))
            purpose = {
                "stage": DockerLookupPurposeV1.RECONCILE_STAGE,
                "submit": DockerLookupPurposeV1.RECONCILE_SUBMIT,
                "cancel": DockerLookupPurposeV1.RECONCILE_CANCEL,
            }[binding.effect_kind]
            if binding.effect_kind == "stage":
                source_request = DockerSourceSealRequestV1(
                    binding.identity, p.roots.source_ref, binding.plan.source_digest
                )
                specialized_request = DockerSourceSealLookupRequestV1(
                    source_request, target.ownership_epoch
                )
                result = self._source_seals.lookup(specialized_request)
            elif binding.effect_kind == "cancel":
                cancellation_request = DockerCancellationRequestV1(
                    binding.identity, binding.cancel_submit_labels,
                    binding.cancel_container_ref, binding.cancel_reason_digest,
                    binding.cancel_authorization_digest,
                )
                specialized_request = DockerCancellationLookupRequestV1(
                    cancellation_request, target.ownership_epoch
                )
                result = self._cancellations.lookup(specialized_request)
            else:
                lookup_request = DockerLookupRequestV1(
                    labels, purpose, target.ownership_epoch
                )
                result = self._control.lookup(lookup_request)
        except Exception:
            return ProviderObservationV1(
                target.effect_id, target.command_digest, self.descriptor.digest,
                ObservationDisposition.INDETERMINATE, target.resolution_digest,
                target.ownership_epoch,
            )
        if binding.effect_kind == "stage":
            return self._stage_lookup_observation(binding, target, specialized_request, result)
        if binding.effect_kind == "cancel":
            return self._cancel_lookup_observation(binding, target, specialized_request, result)
        if type(result) is not DockerLookupResultV1 or result.disposition in {
            DockerLookupDispositionV1.INDETERMINATE, DockerLookupDispositionV1.MULTIPLE
        }:
            disposition = ObservationDisposition.INDETERMINATE
        elif result.disposition is DockerLookupDispositionV1.DEFINITELY_ABSENT:
            absence = result.absence
            try:
                absence_valid = (
                    self._authority.authenticate_absence(absence) is True
                    and (absence.content.request_digest, absence.content.labels_digest,
                         absence.content.purpose, absence.content.generation)
                    == (lookup_request.digest, labels.digest, purpose, target.ownership_epoch)
                )
            except Exception:
                absence_valid = False
            disposition = (ObservationDisposition.DEFINITELY_ABSENT if absence_valid
                           else ObservationDisposition.INDETERMINATE)
        elif result.labels != labels:
            disposition = ObservationDisposition.INDETERMINATE
        else:
            disposition = ObservationDisposition.FOUND
        values = {}
        if disposition is ObservationDisposition.FOUND:
            if binding.effect_kind == "stage":
                values["stage_ref"] = ProviderStageRefV1(
                    self.provider_id, self.profile_ref, self.account_ref,
                    self.namespace_ref, result.container_ref,
                )
            elif binding.effect_kind == "submit":
                values["provider_run"] = ScopedProviderRunRefV1(
                    self.provider_id, self.profile_ref, self.account_ref,
                    self.namespace_ref, result.container_ref,
                )
            else:
                values["cancellation"] = CancellationRefV1(
                    ProviderRunRefV1(binding.cancel_container_ref), binding.cancel_reason_digest
                )
        return ProviderObservationV1(
            binding.effect_id, binding.command_digest, self._profile.executor_descriptor.digest,
            disposition, target.resolution_digest, target.ownership_epoch,
            finality_proof=(result.absence if disposition is ObservationDisposition.DEFINITELY_ABSENT else None),
            **values,
        )

    def _stage_lookup_observation(self, binding, target, lookup_request, result):
        disposition = ObservationDisposition.INDETERMINATE
        values = {}
        proof = None
        try:
            if type(result) is not DockerSourceSealLookupResultV1:
                raise ValueError
            if result.disposition is DockerLookupDispositionV1.FOUND:
                seal = result.seal
                content = _rebuilt(seal.content, DockerSourceSealContentV1)
                request = lookup_request.source_request
                if (self._authority.authenticate_source_seal(seal) is not True
                        or (content.request_digest, content.effect_identity_digest,
                            content.source_ref, content.source_digest, content.read_only)
                        != (request.digest, binding.identity.digest, request.source_ref,
                            request.source_digest, True)):
                    raise ValueError
                disposition = ObservationDisposition.FOUND
                p = binding.plan.profile
                values["stage_ref"] = ProviderStageRefV1(
                    p.provider.provider_id, p.provider.profile_ref,
                    p.scope.account_ref, p.scope.namespace_ref, content.stage_ref,
                )
            elif result.disposition is DockerLookupDispositionV1.DEFINITELY_ABSENT:
                absence = result.absence
                content = (_rebuilt(absence.content, DockerAbsenceContentV1)
                           if type(absence) is AuthenticatedDockerAbsenceV1 else None)
                if (self._authority.authenticate_absence(absence) is not True
                        or (content.request_digest, content.labels_digest,
                            content.purpose, content.generation)
                        != (lookup_request.digest, binding.identity.digest,
                            DockerLookupPurposeV1.RECONCILE_STAGE,
                            target.ownership_epoch)):
                    raise ValueError
                disposition = ObservationDisposition.DEFINITELY_ABSENT
                proof = absence
        except Exception:
            disposition = ObservationDisposition.INDETERMINATE
            values = {}; proof = None
        return ProviderObservationV1(
            binding.effect_id, binding.command_digest,
            binding.plan.profile.executor_descriptor.digest, disposition,
            target.resolution_digest, target.ownership_epoch,
            finality_proof=proof, **values,
        )

    def _cancel_lookup_observation(self, binding, target, lookup_request, result):
        disposition = ObservationDisposition.INDETERMINATE
        values = {}; proof = None
        request = lookup_request.cancellation_request
        try:
            if type(result) is not DockerCancellationLookupResultV1:
                raise ValueError
            if result.disposition is DockerLookupDispositionV1.FOUND:
                evidence = result.evidence
                content = _rebuilt(evidence.content, DockerCancellationContentV1)
                expected = (
                    request.digest, binding.identity.digest,
                    binding.cancel_submit_labels.digest, binding.cancel_container_ref,
                    binding.cancel_reason_digest, binding.cancel_authorization_digest,
                )
                actual = (
                    content.request_digest, content.cancellation_identity_digest,
                    content.submit_labels_digest, content.container_ref,
                    content.reason_digest, content.authorization_digest,
                )
                if self._authority.authenticate_cancellation(evidence) is not True or actual != expected:
                    raise ValueError
                disposition = ObservationDisposition.FOUND
                values["cancellation"] = CancellationRefV1(
                    ProviderRunRefV1(binding.cancel_container_ref),
                    binding.cancel_reason_digest,
                )
            elif result.disposition is DockerLookupDispositionV1.DEFINITELY_ABSENT:
                absence = result.absence
                content = (_rebuilt(absence.content, DockerCancellationAbsenceContentV1)
                           if type(absence) is AuthenticatedDockerCancellationAbsenceV1 else None)
                expected = (
                    lookup_request.digest, request.digest, binding.identity.digest,
                    binding.cancel_authorization_digest,
                    binding.cancel_submit_labels.digest, binding.cancel_container_ref,
                    binding.cancel_reason_digest, target.ownership_epoch,
                )
                actual = (
                    content.lookup_request_digest, content.cancellation_request_digest,
                    content.cancellation_identity_digest, content.authorization_digest,
                    content.submit_labels_digest, content.container_ref,
                    content.reason_digest, content.generation,
                )
                if (self._authority.authenticate_cancellation_absence(absence) is not True
                        or actual != expected
                        or content.resource_phase is not DockerRunPhaseV1.RUNNING):
                    raise ValueError
                disposition = ObservationDisposition.DEFINITELY_ABSENT
                proof = absence
        except Exception:
            disposition = ObservationDisposition.INDETERMINATE
            values = {}; proof = None
        return ProviderObservationV1(
            binding.effect_id, binding.command_digest,
            binding.plan.profile.executor_descriptor.digest, disposition,
            target.resolution_digest, target.ownership_epoch,
            finality_proof=proof, **values,
        )


class DockerReconciliationResolverV1:
    def __init__(self, adapter: DockerReconciliationAdapterV1):
        self._adapter = adapter

    def resolve(self, request: ReconciliationResolutionRequestV2):
        return mint_resolved_adapter(request, self._adapter)


__all__: list[str] = []
