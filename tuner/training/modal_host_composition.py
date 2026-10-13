"""Provider-free assembly of the first same-process packaged Modal API host."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from tuner.training.modal_host_authority import (
    HMACAuthenticator, LogAuthenticator, ObservationAuthenticator,
    ReaderEvidenceAuthority,
)
from synaptic_tuner.api.v1.providers import (
    ProviderCapabilities, ProviderDescriptor, ProviderRef,
)
from synaptic_tuner.api.v1.reference.authority import compose_reference_authority
from synaptic_tuner.api.v1.reference.composition import compose_reference_host
from synaptic_tuner.api.v1.reference.provider_family import (
    ProviderFamilyV1, ReferenceHostPortsV1, require_methods,
)
from synaptic_tuner.api.v1.reference.stores import (
    InMemoryDurableRecordStoreV1, InMemoryDurableStreamStoreV1,
)
from synaptic_tuner.api.v1.reference.training import ReferenceRequestPortsV1
from synaptic_tuner.api.v1.secrets import SecretRef
from tuner.execution.coordinator_v1.model import ProviderExecutionBindingV1
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest
from tuner.execution.foundation_v2.references import ExecutionScopeV1
from tuner.execution.providers.modal.packaged_composition import ModalPackagedComposition
from tuner.training.modal_host_artifacts import ModalPackagedArtifactVerifierV1
from tuner.training.modal_host_reader import ModalPackagedCoordinatorReaderV1
from tuner.training.modal_host_requests import (
    ModalPackagedPlanPortsV1, ModalPackagedRequestsV1,
)


@dataclass(frozen=True, slots=True)
class ModalPackagedReferenceHostV1:
    composition: object
    adapter: ModalPackagedComposition
    effects: object
    reader: ModalPackagedCoordinatorReaderV1
    verifier: ModalPackagedArtifactVerifierV1

    @property
    def api(self):
        return self.composition.api()


def _transport_logs_authorized(qualification, effects, components):
    """Only an exact same-process, durably retained CPU transport proof grants logs."""
    if qualification is None:
        return False
    from tuner.training.modal_host_qualification import ModalHostCPUQualificationV1
    from tuner.training.modal_host_effects import ModalPackagedHostEffectsV1
    if type(qualification) is not ModalHostCPUQualificationV1:
        raise ValueError("exact CPU qualification required")
    if qualification.transport_verification is None:
        return False
    from tuner.execution.providers.modal.runtime_release_qualification_reader import ModalCPUTransportVerificationV1
    from tuner.execution.providers.modal.runtime_release_qualification import (
        ModalCPUTransportProbeV1, ModalRuntimeReleaseQualificationReceiptV1,
    )
    from tuner.execution.providers.modal.runtime_release_deployment import ModalRuntimeReleaseDeploymentFactsV1
    from tuner.execution.providers.modal.packaged_binding import ModalPackagedRuntimeFactsV1
    proof = qualification.transport_verification
    if type(proof) is not ModalCPUTransportVerificationV1:
        raise ValueError("exact CPU transport verification required")
    document = proof.to_dict()
    if document.get("status") != "passed" or document.get("durability_acknowledged") is not True:
        return False
    if (type(effects) is not ModalPackagedHostEffectsV1
            or type(qualification.receipt) is not ModalRuntimeReleaseQualificationReceiptV1
            or type(qualification.deployment_facts) is not ModalRuntimeReleaseDeploymentFactsV1
            or type(qualification.transport_probe) is not ModalCPUTransportProbeV1):
        raise ValueError("CPU transport composition binding invalid")
    receipt, deployment = qualification.receipt, qualification.deployment_facts
    if (deployment.schema_version != "synaptic-modal-runtime-release-deployment-facts/v2"
            or effects._facts != ModalPackagedRuntimeFactsV1.from_release_deployment(deployment)
            or qualification.runtime_release_digest != effects._release.manifest_digest
            or components.execution_source.runtime_release_digest != effects._release.manifest_digest
            or qualification.deployment_facts_digest != deployment.facts_digest
            or receipt.deployment_facts_digest != deployment.facts_digest
            or qualification.provider_call_id != receipt.provider_call_id
            or qualification.output_sha256 != receipt.output_sha256
            or qualification.training_executed is not False or qualification.gpu_qualified is not False
            or document.get("core_receipt_sha256") != hashlib.sha256(receipt.canonical_bytes).hexdigest()
            or document.get("dispatch_digest") != receipt.dispatch_digest
            or document.get("provider_call_id") != receipt.provider_call_id
            or document.get("runtime_release_digest") != effects._release.manifest_digest
            or document.get("deployment_facts_digest") != deployment.facts_digest
            or document.get("probe") != qualification.transport_probe.to_dict()
            or document.get("capture_sha256") != qualification.transport_capture_sha256
            or document.get("training_executed") is not False or document.get("gpu_qualified") is not False):
        raise ValueError("CPU transport composition binding invalid")
    return True


def _modal_packaged_descriptor(*, logs: bool):
    """The two existing descriptor identities; this is not a log proof."""
    if type(logs) is not bool:
        raise TypeError("exact descriptor capability required")
    return ProviderDescriptor(
        "synaptic-provider-descriptor/v1", "modal", "Modal packaged training",
        "0.1.0", ProviderCapabilities(True, logs, False, False, True, False),
    )


def _modal_read_authorities(*, key, clock, bindings, binding_authority,
                            foundation_authenticator, assessment_authenticator,
                            packaged_reader, facade):
    reader_key_ref = "modal-packaged-reader-key-v1"
    evidence = ReaderEvidenceAuthority(
        "modal-packaged-reader", reader_key_ref,
        HMACAuthenticator(
            {reader_key_ref: key},
            allowed_purposes=frozenset({
                "provider-run-observation/v1", "provider-log-page/v1",
            }),
        ),
    )
    reader = ModalPackagedCoordinatorReaderV1(
        bindings=bindings, binding_authority=binding_authority,
        foundation_repository=None,
        foundation_authenticator=foundation_authenticator,
        assessment_authenticator=assessment_authenticator,
        packaged_reader=packaged_reader,
        facade=facade,
        evidence_authority=evidence, clock=clock,
    )
    verifier = ModalPackagedArtifactVerifierV1(
        reader=reader,
        foundation_authenticator=foundation_authenticator,
        assessment_authenticator=assessment_authenticator,
        authority_ref="modal-packaged-artifact-verifier",
        key_ref="modal-packaged-artifact-key-v1",
        key=hashlib.sha256(b"modal-packaged-artifact-key/v1\0" + key).digest(),
        clock=clock,
    )
    return evidence, reader, verifier


def _compose_modal_packaged_reference_host(
    *, adapter: ModalPackagedComposition, effects: object,
    prepared_request: object, run: object, components: object,
    context: object, input_preparation: object, clock: object,
    secrets: object, authority_secret: SecretRef, reader_secret: SecretRef,
    profile_digest: str, quote_digest: str,
    maximum_cost_minor_units: int,
    qualification_admission=None,
    cpu_qualification=None,
    records=None, streams=None,
) -> ModalPackagedReferenceHostV1:
    """Compose only from already-captured and admitted packaged material.

    Build/deploy/quote are separate explicit host effects before this call.
    No provider calls or catalog reads occur during this assembly. Optional
    paired stores remain host-owned; absent stores preserve in-memory defaults.
    """
    if type(adapter) is not ModalPackagedComposition:
        raise TypeError("exact packaged Modal adapter required")
    if (records is None) != (streams is None):
        raise TypeError("record and stream stores must be supplied together")
    if records is not None:
        require_methods(records, "create", "read", "compare_and_swap", "put_if_absent", "list_page")
        require_methods(streams, "append", "read_page")
    else:
        records, streams = InMemoryDurableRecordStoreV1(), InMemoryDurableStreamStoreV1()
    if not all(callable(getattr(effects, name, None)) for name in (
        "authorize", "bind", "authenticate",
    )):
        raise TypeError("complete packaged host effect authority required")
    raw_reader_key = secrets.resolve(reader_secret)
    if type(raw_reader_key) is not str or len(raw_reader_key) < 32:
        raise ValueError("named reader authority secret is unavailable")
    key = raw_reader_key.encode("utf-8")
    del raw_reader_key
    if qualification_admission is not None:
        from tuner.training.modal_metrics_qualification import MetricsQualificationAdmission, current_clean_commit
        from tuner.training.modal_host_effects import ModalPackagedHostEffectsV1
        from tuner.training.packaged_compilation import compile_packaged_sft_workload, packaged_configuration_digest
        from tuner.runtime_profiles import load_runtime_profile
        from tuner.training.modal_recipe import resolve_modal_sft_build
        if (type(qualification_admission) is not MetricsQualificationAdmission
                or type(effects) is not ModalPackagedHostEffectsV1):
            raise ValueError("metrics_qualification_binding_invalid")
        admitted = qualification_admission.document
        execution = components.execution_source
        model = components.resolved_config.to_dict()["model"]
        profiles_root = context.engine_root / "Trainers" / "runtime_profiles"
        selected = load_runtime_profile(adapter.effect_executor.profile_ref, profiles_root).resolve(
            model=model["ref"], model_revision=model["revision"], method="sft",
        )
        _, current_intent = resolve_modal_sft_build(selected, profiles_root, model["ref"], model["revision"])
        qualification_admission.validate_scope(effects._facts)
        if (current_clean_commit(context.engine_root) != admitted["engine_commit"]
                or current_intent["intent_digest"] != admitted["runtime_material_intent_digest"]
                or selected.profile_sha256.removeprefix("sha256:") != admitted["profile_sha256"]
                or effects._storage.attempts.resolve("metrics-qual-" + admitted["attempt_ref"]) != qualification_admission.canonical
                or effects._execution != execution or effects._execution.run_ref != run.run_id
                or effects._workload != compile_packaged_sft_workload(resolved_config=components.resolved_config).canonical_bytes
                or execution.workload_digest.removeprefix("sha256:") != admitted["workload_digest"]
                or packaged_configuration_digest(components.resolved_config).removeprefix("sha256:") != admitted["configuration_digest"]
                or execution.prepared_input_revision.removeprefix("sha256:") != admitted["dataset_digest"]
                or execution.prepared_input_content_digest.removeprefix("sha256:") != admitted["dataset_content_digest"]
                or profile_digest != admitted["profile_sha256"]
                or components.resources.accelerator != admitted["accelerator"]
                or components.resources.accelerator_count != admitted["accelerator_count"]
                or components.resources.timeout_seconds != admitted["timeout_seconds"]
                or maximum_cost_minor_units != admitted["maximum_cost_minor_units"]):
            raise ValueError("metrics_qualification_binding_invalid")
    descriptor = _modal_packaged_descriptor(logs=qualification_admission is not None or
        _transport_logs_authorized(cpu_qualification, effects, components))
    executor = adapter.effect_executor
    reconciliation = adapter.reconciliation_adapter
    provider = ProviderRef("modal", executor.profile_ref)
    scope = ExecutionScopeV1(executor.account_ref, executor.namespace_ref)
    resources = components.resources
    resource_digest = domain_digest(
        "synaptic-modal-packaged-resource-request/v1", canonical_bytes({
            "accelerator": resources.accelerator,
            "accelerator_count": resources.accelerator_count,
            "timeout_seconds": resources.timeout_seconds,
        }),
    )
    secret_requirements_digest = domain_digest(
        "synaptic-modal-packaged-secret-requirements/v1", canonical_bytes({
            "authority": authority_secret.to_dict(),
            "reader": reader_secret.to_dict(),
        }),
    )
    binding = ProviderExecutionBindingV1(
        provider, descriptor.descriptor_digest, profile_digest, scope,
        executor.descriptor, reconciliation.descriptor.digest,
        resource_digest, quote_digest, secret_requirements_digest,
    )
    planning = ModalPackagedPlanPortsV1(
        provider=provider, descriptor=descriptor, binding=binding,
        clock=clock, maximum_cost_minor_units=maximum_cost_minor_units,
    )
    request_ports = ModalPackagedRequestsV1(
        prepared_request=prepared_request, run=run,
        components=components, context=context,
    )
    ports = ReferenceHostPortsV1(
        records, streams, clock, effects, secrets,
    )
    authority = compose_reference_authority(
        clock=clock, secrets=secrets, authority_secret=authority_secret,
    )
    evidence, reader, verifier = _modal_read_authorities(
        key=key, clock=authority.clock, bindings=effects.bindings, binding_authority=effects,
        foundation_authenticator=authority.foundation_authenticator,
        assessment_authenticator=authority.assessment_authority,
        packaged_reader=adapter.reader, facade=adapter.reader._facade,
    )
    family = ProviderFamilyV1(
        descriptor, planning, planning, reader,
        adapter.executor_resolver, adapter.reconciliation_resolver,
        evidence, verifier, ObservationAuthenticator(evidence),
        LogAuthenticator(evidence),
    )
    composed = compose_reference_host(
        family=family, ports=ports,
        requests=ReferenceRequestPortsV1(request_ports, request_ports, request_ports),
        authority=authority, input_preparation=input_preparation,
    )
    reader.bind_foundation(composed.foundation)
    verifier.bind_foundation(composed.foundation)
    return ModalPackagedReferenceHostV1(composed, adapter, effects, reader, verifier)


def compose_modal_packaged_reference_host(**kwargs):
    """Ordinary logs require a matching, verified and durably retained CPU proof."""
    if "qualification_admission" in kwargs:
        raise TypeError("qualification authority requires the restricted composition")
    return _compose_modal_packaged_reference_host(**kwargs)


def compose_modal_metrics_qualification_host(*, admission, **kwargs):
    """One admitted, journal-claimed source/run/scope; no ordinary capability change."""
    if admission is None or "qualification_admission" in kwargs:
        raise TypeError("exact metrics qualification admission required")
    return _compose_modal_packaged_reference_host(qualification_admission=admission, **kwargs)
