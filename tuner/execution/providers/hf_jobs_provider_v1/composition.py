"""Ordinary executable HF family over the existing host authority and stores."""

import base64
import json

from synaptic_tuner.api.v1.reference import compose_reference_host
from synaptic_tuner.api.v1.reference.provider_family import ProviderFamilyV1, ReferenceHostPortsV1
from tuner.execution.foundation_v2.commands import parse_exact_command, CancelCommandV2
from tuner.execution.foundation_v2.executors import mint_resolved_executor, mint_resolved_adapter
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.foundation_v2.observations import ProviderObservationV1, ObservationDisposition
from tuner.execution.foundation_v2.references import (
    ProviderStageRefV1, ScopedProviderRunRefV1, CancellationRefV1, ProviderRunRefV1,
)
from tuner.execution.foundation_v2.reconciliation import ReconciliationTargetV1
from tuner.execution.coordinator_v1.model import (
    AuthenticatedFoundationRecordAssessmentV1, FoundationDispositionV1,
)
from tuner.execution.coordinator_v1.state_machine import _derive_foundation
from synaptic_tuner.api.v1.results import TrainingRunRef
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
    effect_kinds = ("stage", "submit", "cancel")
    payload_schemas = ("stage-payload/v2", "submit-payload/v2", "cancel-payload/v2")

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
            elif command.operation.effect.kind is EffectKind.CANCEL:
                result = self.transport.cancel(command)
                values["cancellation"] = CancellationRefV1(
                    ProviderRunRefV1(result["job_id"]), result["reason_digest"])
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


class _Reconciliation:
    capabilities = ("lookup",)

    def __init__(self, preparation, transport):
        self.descriptor = preparation.adapter
        self.transport = transport
        s = preparation.selection
        self.provider_id, self.profile_ref = s.provider.provider_id, s.provider.profile_ref
        self.account_ref, self.namespace_ref = s.scope.account_ref, s.scope.namespace_ref

    def lookup(self, target, preparation):
        if type(target) is not ReconciliationTargetV1:
            raise HFJobsError("binding_mismatch")
        command = parse_exact_command(target.command_bytes)
        if (command.digest != target.command_digest
                or command.operation.effect.effect_id != target.effect_id
                or command.preparation != preparation):
            raise HFJobsError("binding_mismatch")
        catalog = self.transport.retained("command", command.digest)
        if base64.b64decode(catalog["command_base64"], validate=True) != command.canonical_bytes:
            raise HFJobsError("binding_mismatch")
        values = {}
        try:
            result = self.transport.lookup(command)
            if result is None:
                disposition = ObservationDisposition.INDETERMINATE
            else:
                kind = command.operation.effect.kind
                if kind is EffectKind.STAGE:
                    values["stage_ref"] = ProviderStageRefV1(self.provider_id, self.profile_ref,
                        self.account_ref, self.namespace_ref, result["stage_ref"])
                elif kind is EffectKind.SUBMIT:
                    values["provider_run"] = ScopedProviderRunRefV1(self.provider_id, self.profile_ref,
                        self.account_ref, self.namespace_ref, result["job_id"])
                else:
                    values["cancellation"] = CancellationRefV1(
                        ProviderRunRefV1(result["job_id"]), result["reason_digest"])
                disposition = ObservationDisposition.FOUND
        except Exception:
            disposition = ObservationDisposition.INDETERMINATE
            values = {}
        return ProviderObservationV1(command.operation.effect.effect_id, command.digest,
            command.executor.digest, disposition, target.resolution_digest,
            target.ownership_epoch, **values)


class _CancelAccess:
    """Read original authenticated workflow/submit on the host owning thread."""

    def __init__(self, composition, preparation, authority, transport):
        self.composition, self.preparation, self.authority = composition, preparation, authority
        self.transport = transport

    def __call__(self, command):
        if type(command) is not CancelCommandV2:
            raise HFJobsError("binding_mismatch")
        prep = command.preparation
        workflow = self.composition.stores.workflow_store.get(TrainingRunRef(prep.run_id, prep.project_ref))
        if (workflow is None or workflow.plan_fingerprint != self.preparation.plan.plan_fingerprint
                or workflow.preparation_digest != prep.preparation_digest
                or workflow.provider_context_digest != self.preparation.context_value.provider_context_digest
                or workflow.provider_descriptor_digest != self.preparation.descriptor.descriptor_digest
                or workflow.cancel is None or workflow.cancel.canonical_command_bytes != command.canonical_bytes
                or workflow.submit is None or workflow.provider_run_ref is None
                or not workflow.submit.foundation_bindings or not workflow.submit.foundation_outcomes):
            raise HFJobsError("binding_mismatch")
        submit = workflow.submit
        record = self.composition.foundation.get(submit.effect_id)
        assessment = AuthenticatedFoundationRecordAssessmentV1.parse(
            submit.foundation_bindings[-1].canonical_assessment_bytes)
        derived = _derive_foundation(submit, record, assessment,
            self.authority.foundation_authenticator, self.authority.assessment_authority, None)
        if (derived != (submit.foundation_bindings[-1], submit.foundation_outcomes[-1], workflow.provider_run_ref)
                or derived[1].disposition is not FoundationDispositionV1.FOUND
                or derived[2].reference.provider_job_ref != command.to_dict()["cancellation"]["provider_job_ref"]):
            raise HFJobsError("binding_mismatch")
        original = parse_exact_command(submit.canonical_command_bytes)
        catalog = self.transport.retained("command", original.digest)
        if base64.b64decode(catalog["command_base64"], validate=True) != original.canonical_bytes:
            raise HFJobsError("binding_mismatch")
        return original


class _ReconciliationResolver:
    def __init__(self, adapter):
        self.adapter = adapter

    def resolve(self, request):
        return mint_resolved_adapter(request, self.adapter)


def compose_hf_jobs_host(*, preparation, sdk, api, ports, requests, authority,
                         scratch_root, reader_authority_ref, reader_key_ref,
                         reader_key, deadline_seconds=30,
                         allow_current_cancel_authorization=False):
    """Return the ordinary reference host, not a separate HF lifecycle controller.

    Host supplies its existing stores, authority, grants, credentials and request
    resolution. Cancellation and retained-receipt lookup machinery use the
    existing Foundation/store ports, while descriptor capabilities stay disabled
    by default. Current cancellation authorization is an explicit host opt-in;
    source-free cold reconstruction and live qualification remain separate.
    Logs and restart continuation remain unavailable.
    """
    if (type(ports) is not ReferenceHostPortsV1
            or type(allow_current_cancel_authorization) is not bool):
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
        _ExecutorResolver(executor), _ReconciliationResolver(_Reconciliation(preparation, transport)),
        evidence, verifier, ObservationAuthenticator(evidence), LogAuthenticator(evidence))
    composition = compose_reference_host(family=family,
        ports=ReferenceHostPortsV1(ports.records, ports.streams, ports.clock,
                                  _BoundGrants(ports.grants, transport), ports.secrets),
        requests=requests, authority=authority,
        allow_current_cancel_authorization=allow_current_cancel_authorization)
    transport.bind_cancel_access(_CancelAccess(composition, preparation, authority, transport))
    verifier.bind_foundation(composition.foundation)
    return composition


__all__: list[str] = []
