"""Ordinary executable HF family over the existing host authority and stores."""

import base64
import json

from synaptic_tuner.api.v1.reference import compose_reference_host
from synaptic_tuner.api.v1.reference.provider_family import ProviderFamilyV1, ReferenceHostPortsV1
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.foundation_v2.executors import mint_resolved_executor, mint_resolved_adapter
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.foundation_v2.observations import ProviderObservationV1, ObservationDisposition
from tuner.execution.foundation_v2.references import ProviderStageRefV1, ScopedProviderRunRefV1
from tuner.training.modal_host_authority import (
    ReaderEvidenceAuthority, HMACAuthenticator, ObservationAuthenticator, LogAuthenticator,
)
from .model import HFJobsError, document_bytes
from .transport import HFJobsTransport
from .reader import HFJobsReader, HFJobsArtifactVerifier


class _BoundGrants:
    """Delegate host permission, retaining its exact command for digest-only dispatch."""

    def __init__(self, delegate, transport):
        self.delegate, self.transport = delegate, transport

    def authorize(self, requirements):
        return self.delegate.authorize(requirements)

    def bind(self, grant, *, operation, requirements):
        command = parse_exact_command(operation.canonical_bytes)
        self.transport._validate(command, command.operation.effect.kind)
        permitted = self.delegate.bind(grant, operation=operation, requirements=requirements)
        raw = document_bytes({"schema_version": "synaptic-hf-command-catalog/v1",
            "command_digest": command.digest, "material_digest": self.transport.preparation.material_digest,
            "command_base64": base64.b64encode(command.canonical_bytes).decode("ascii")})
        if self.transport.records.put_if_absent(partition="effects",
            key=self.transport._key("command", command.digest), canonical=raw) is not True:
            raise HFJobsError("binding_mismatch")
        return permitted


class _Executor:
    effect_kinds = ("stage", "submit")
    payload_schemas = ("stage-payload/v2", "submit-payload/v2")

    def __init__(self, preparation, transport, clock):
        self.preparation, self.transport, self.clock = preparation, transport, clock
        self.descriptor = preparation.executor
        s = preparation.selection
        self.provider_id, self.profile_ref = s.provider.provider_id, s.provider.profile_ref
        self.account_ref, self.namespace_ref = s.scope.account_ref, s.scope.namespace_ref

    def execute_once(self, payload, request):
        retained = self.transport.records.read(partition="effects",
            key=self.transport._key("command", request.command_digest))
        if retained is None:
            raise HFJobsError("binding_mismatch")
        document = json.loads(retained.canonical)
        command = parse_exact_command(base64.b64decode(document["command_base64"], validate=True))
        if (command.digest != request.command_digest or command.payload != payload
                or document["material_digest"] != self.preparation.material_digest
                or request.descriptor_digest != self.descriptor.digest
                or (request.provider_id, request.profile_ref, request.account_ref, request.namespace_ref)
                != (self.provider_id, self.profile_ref, self.account_ref, self.namespace_ref)
                or request.effect_kind != command.operation.effect.kind.value
                or request.payload_schema != command.payload.payload_kind
                or request.input_digest != command.payload.input_digest):
            raise HFJobsError("binding_mismatch")
        self.transport._validate(command, command.operation.effect.kind)
        values = {}
        try:
            if command.operation.effect.kind is EffectKind.STAGE:
                result = self.transport.stage(command)
                values["stage_ref"] = ProviderStageRefV1(self.provider_id, self.profile_ref,
                    self.account_ref, self.namespace_ref, result["stage_ref"])
            elif command.operation.effect.kind is EffectKind.SUBMIT:
                result = self.transport.submit(command)
                values["provider_run"] = ScopedProviderRunRefV1(self.provider_id, self.profile_ref,
                    self.account_ref, self.namespace_ref, result["job_id"])
            else:
                raise HFJobsError("capability_unavailable")
            disposition = ObservationDisposition.FOUND
        except Exception:
            # No exception text or raw SDK payload becomes provider evidence.
            # A partial upload or unknown job submission cannot be replayed.
            disposition = ObservationDisposition.INDETERMINATE
            values = {}
        return ProviderObservationV1(command.operation.effect.effect_id, command.digest,
            command.executor.digest, disposition, request.digest, 1, **values)


class _ExecutorResolver:
    def __init__(self, executor):
        self.executor = executor

    def resolve(self, request):
        return mint_resolved_executor(request, self.executor)


class _UnavailableReconciliation:
    capabilities = ("lookup",)

    def __init__(self, preparation):
        self.descriptor = preparation.adapter
        s = preparation.selection
        self.provider_id, self.profile_ref = s.provider.provider_id, s.provider.profile_ref
        self.account_ref, self.namespace_ref = s.scope.account_ref, s.scope.namespace_ref

    def lookup(self, target, preparation):
        raise HFJobsError("capability_unavailable")


class _ReconciliationResolver:
    def __init__(self, adapter):
        self.adapter = adapter

    def resolve(self, request):
        return mint_resolved_adapter(request, self.adapter)


def compose_hf_jobs_host(*, preparation, sdk, api, ports, requests, authority,
                         scratch_root, reader_authority_ref, reader_key_ref,
                         reader_key, deadline_seconds=30):
    """Return the ordinary reference host, not a separate HF lifecycle controller.

    Host supplies its existing stores, authority, grants, credentials and request
    resolution. Family supports SFT start/observe/artifacts; cancellation,
    reconciliation, logs and restart continuation remain unavailable.
    """
    if type(ports) is not ReferenceHostPortsV1:
        raise HFJobsError("input_invalid")
    transport = HFJobsTransport(preparation=preparation, sdk=sdk, api=api,
        records=ports.records, secrets=ports.secrets, scratch_root=scratch_root,
        deadline_seconds=deadline_seconds)
    evidence = ReaderEvidenceAuthority(reader_authority_ref, reader_key_ref,
        HMACAuthenticator({reader_key_ref: reader_key}, allowed_purposes=frozenset({
            "provider-run-observation/v1", "provider-log-page/v1"})))
    reader = HFJobsReader(preparation=preparation, transport=transport,
        foundation_authenticator=authority.foundation_authenticator,
        assessment_authenticator=authority.assessment_authority, evidence_authority=evidence,
        clock=authority.clock)
    verifier = HFJobsArtifactVerifier(reader=reader, authority_ref=reader_authority_ref,
        key_ref=reader_key_ref, key=reader_key, clock=authority.clock)
    executor = _Executor(preparation, transport, authority.clock)
    family = ProviderFamilyV1(preparation.descriptor, preparation, preparation, reader,
        _ExecutorResolver(executor), _ReconciliationResolver(_UnavailableReconciliation(preparation)),
        evidence, verifier, ObservationAuthenticator(evidence), LogAuthenticator(evidence))
    composition = compose_reference_host(family=family,
        ports=ReferenceHostPortsV1(ports.records, ports.streams, ports.clock,
                                  _BoundGrants(ports.grants, transport), ports.secrets),
        requests=requests, authority=authority)
    verifier.bind_foundation(composition.foundation)
    return composition


__all__: list[str] = []
