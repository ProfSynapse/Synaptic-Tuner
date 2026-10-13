"""Caller-stored one-shot Docker stops using existing cancellation evidence.

The caller retains the record store and signing authority. Missing or uncertain
receipts never authorize another stop. A receipt attests the successful stop
request and observed nonrunning state; natural completion can race that request.
The session deadline bounds CLI/provider work. Caller-owned synchronous signing,
authentication and store calls retain their own time semantics; this wrapper
does not preempt them or expire historical receipts with the old CLI deadline.
"""
from __future__ import annotations

from synaptic_tuner.api.v1.ports import StoragePartition, StoredRecordV1
from synaptic_tuner.api.v1.reference.repositories import _decode, _encode
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest, parse_canonical_object
from .cli_transport import DockerCLISessionV1
from .model import (
    AuthenticatedDockerCancellationEvidenceV1, DockerCancellationContentV1,
    DockerCancellationLookupRequestV1, DockerCancellationLookupResultV1,
    DockerCancellationRequestV1, DockerDiagnosticCodeV1, DockerLookupDispositionV1,
    DockerProviderError,
)

_PARTITION = StoragePartition.EFFECTS.value
_MAX_RECORD_BYTES = 16384


class DockerCancellationTransportV1:
    """Existing cancellation port, composed explicitly with caller-owned storage."""

    def __init__(self, *, session, records, issue_evidence, evidence_authority,
                 grace_seconds):
        if (type(session) is not DockerCLISessionV1 or type(grace_seconds) is not int
                or not 1 <= grace_seconds <= 10 or not callable(issue_evidence)
                or not callable(getattr(evidence_authority, 'authenticate_cancellation', None))
                or any(not callable(getattr(records, method, None))
                       for method in ('read', 'create', 'compare_and_swap'))):
            raise TypeError('docker_cancellation_configuration_invalid')
        self._session, self._records = session, records
        self._issue, self._authority = issue_evidence, evidence_authority
        self._grace = grace_seconds

    @staticmethod
    def _key(request):
        identity = request.cancellation_identity
        profile = identity.plan.profile
        digest = domain_digest('synaptic-docker-cancellation-slot/v1', canonical_bytes({
            'provider_id': profile.provider.provider_id, 'profile_ref': profile.provider.profile_ref,
            'account_ref': profile.scope.account_ref, 'namespace_ref': profile.scope.namespace_ref,
            'effect_id': identity.effect_id,
        }))
        return 'docker-cancellation/' + digest

    def _request(self, request):
        if type(request) is not DockerCancellationRequestV1:
            raise ValueError
        # Rebuild through the existing strict generic codec, including retained
        # profile validation; never trust mutation of a frozen object's internals.
        rebuilt = _decode(_encode(request, DockerCancellationRequestV1), DockerCancellationRequestV1)
        if (rebuilt != request or (self._session._profile is not None
                and rebuilt.cancellation_identity.plan.profile != self._session._profile)):
            raise ValueError
        return rebuilt

    @staticmethod
    def _document(request, evidence):
        return canonical_bytes({'request_digest': request.digest,
            'evidence': None if evidence is None else _encode(evidence, AuthenticatedDockerCancellationEvidenceV1)})

    def _evidence(self, request, evidence):
        if type(evidence) is not AuthenticatedDockerCancellationEvidenceV1:
            raise ValueError
        rebuilt = _decode(_encode(evidence, AuthenticatedDockerCancellationEvidenceV1), AuthenticatedDockerCancellationEvidenceV1)
        content = rebuilt.content
        if ((content.request_digest, content.cancellation_identity_digest, content.submit_labels_digest,
             content.container_ref, content.reason_digest, content.authorization_digest)
                != (request.digest, request.cancellation_identity.digest, request.submit_labels.digest,
                    request.container_ref, request.reason_digest, request.authorization_digest)
                or self._authority.authenticate_cancellation(rebuilt) is not True):
            raise ValueError
        return rebuilt

    def _read(self, request):
        key = self._key(request)
        stored = self._records.read(partition=_PARTITION, key=key)
        if stored is None:
            return None, None
        if type(stored) is not StoredRecordV1 or stored.key != key or len(stored.canonical) > _MAX_RECORD_BYTES:
            raise ValueError
        document = parse_canonical_object(stored.canonical, name='Docker cancellation record')
        if (set(document) != {'request_digest', 'evidence'} or document['request_digest'] != request.digest
                or canonical_bytes(document) != stored.canonical):
            raise ValueError
        raw = document['evidence']
        evidence = None if raw is None else self._evidence(request, _decode(raw, AuthenticatedDockerCancellationEvidenceV1))
        return stored, evidence

    def stop_once(self, request):
        try:
            request = self._request(request)
            stored, evidence = self._read(request)
            if stored is not None:
                if evidence is None:
                    raise ValueError
                return evidence
            claim = self._document(request, None)
            accepted = self._records.create(partition=_PARTITION, key=self._key(request), canonical=claim)
            if accepted is not True:
                # A concurrent winner may still be stopping. Lookup only; never
                # wait for it, assume absence or use a split/retry fallback.
                if accepted is not False:
                    raise ValueError
                _, evidence = self._read(request)
                if evidence is None:
                    raise ValueError
                return evidence
            stored, evidence = self._read(request)
            if stored is None or stored.canonical != claim or evidence is not None:
                raise ValueError
            observation = self._session._stop_exact(request.container_ref, request.submit_labels,
                request.cancellation_identity.plan.profile.image, grace_seconds=self._grace)
            content = DockerCancellationContentV1(
                request.digest, request.cancellation_identity.digest, request.submit_labels.digest,
                request.container_ref, request.reason_digest, request.authorization_digest,
                domain_digest('synaptic-docker-stop-observation/v1', canonical_bytes(observation)),
            )
            evidence = self._evidence(request, self._issue(content))
            if evidence.content != content:
                raise ValueError
            accepted = self._records.compare_and_swap(partition=_PARTITION, key=self._key(request),
                expected_revision=stored.revision, canonical=self._document(request, evidence))
            if accepted is not True:
                raise ValueError
            return evidence
        except Exception:
            raise DockerProviderError(DockerDiagnosticCodeV1.STOP_INDETERMINATE) from None

    def lookup(self, request):
        try:
            if type(request) is not DockerCancellationLookupRequestV1:
                raise ValueError
            request = DockerCancellationLookupRequestV1(self._request(request.cancellation_request), request.generation)
            _, evidence = self._read(request.cancellation_request)
            if evidence is not None:
                return DockerCancellationLookupResultV1(DockerLookupDispositionV1.FOUND, evidence=evidence)
        except Exception:
            pass
        return DockerCancellationLookupResultV1(DockerLookupDispositionV1.INDETERMINATE)
