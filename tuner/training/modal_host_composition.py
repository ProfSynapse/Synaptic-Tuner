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
    ProviderFamilyV1, ReferenceHostPortsV1,
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


def compose_modal_packaged_reference_host(
    *, adapter: ModalPackagedComposition, effects: object,
    prepared_request: object, run: object, components: object,
    context: object, input_preparation: object, clock: object,
    secrets: object, authority_secret: SecretRef, reader_secret: SecretRef,
    profile_digest: str, quote_digest: str,
    maximum_cost_minor_units: int,
) -> ModalPackagedReferenceHostV1:
    """Compose only from already-captured and admitted packaged material.

    Build/deploy/quote are separate explicit host effects before this call.
    No provider calls or catalog reads occur during this assembly.
    """
    if type(adapter) is not ModalPackagedComposition:
        raise TypeError("exact packaged Modal adapter required")
    if not all(callable(getattr(effects, name, None)) for name in (
        "authorize", "bind", "authenticate",
    )):
        raise TypeError("complete packaged host effect authority required")
    raw_reader_key = secrets.resolve(reader_secret)
    if type(raw_reader_key) is not str or len(raw_reader_key) < 32:
        raise ValueError("named reader authority secret is unavailable")
    key = raw_reader_key.encode("utf-8")
    del raw_reader_key
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
    descriptor = ProviderDescriptor(
        "synaptic-provider-descriptor/v1", "modal", "Modal packaged training",
        "0.1.0", ProviderCapabilities(True, False, False, False, True, False),
    )
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
        InMemoryDurableRecordStoreV1(), InMemoryDurableStreamStoreV1(),
        clock, effects, secrets,
    )
    authority = compose_reference_authority(
        clock=clock, secrets=secrets, authority_secret=authority_secret,
    )
    reader = ModalPackagedCoordinatorReaderV1(
        bindings=effects.bindings, binding_authority=effects,
        foundation_repository=None,
        foundation_authenticator=authority.foundation_authenticator,
        assessment_authenticator=authority.assessment_authority,
        packaged_reader=adapter.reader,
        facade=adapter.reader._facade,
        evidence_authority=evidence, clock=authority.clock,
    )
    verifier = ModalPackagedArtifactVerifierV1(
        reader=reader,
        foundation_authenticator=authority.foundation_authenticator,
        assessment_authenticator=authority.assessment_authority,
        authority_ref="modal-packaged-artifact-verifier",
        key_ref="modal-packaged-artifact-key-v1",
        key=hashlib.sha256(b"modal-packaged-artifact-key/v1\0" + key).digest(),
        clock=authority.clock,
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
