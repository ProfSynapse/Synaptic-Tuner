"""One provider-neutral deterministic fake family for internal conformance tests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
from threading import Lock

from synaptic_tuner.api.v1.providers import ProviderDescriptor, ProviderRef
from synaptic_tuner.api.v1.results import TrainingRunRef, VerifiedArtifact
from synaptic_tuner.api.v1.runs_facade import RunLogEntry

from .foundation_v2.canonical import canonical_bytes, digest_text, domain_digest, safe_ref
from .foundation_v2.executors import (
    AdapterDescriptorV1,
    ExecutionResolutionRequestV2,
    ExecutorDescriptorV1,
    ReconciliationResolutionRequestV2,
    mint_resolved_adapter,
    mint_resolved_executor,
)
from .foundation_v2.identities import EffectKind
from .foundation_v2.observations import ObservationDisposition, ProviderObservationV1
from .foundation_v2.references import (
    CancellationRefV1,
    ProviderRunRefV1,
    ProviderStageRefV1,
    ScopedProviderRunRefV1,
)
from .foundation_v2.registry import (
    ProviderReaderFactoryRequestV1,
    ResolvedProviderReaderV1,
)
from .coordinator_v1.model import (
    ArtifactManifestV1,
    ArtifactVerificationContentV1,
    AuthenticatedArtifactVerificationReceiptV1,
    AuthenticatedProviderLogPageV1,
    AuthenticatedProviderRunObservationV1,
    ProviderLogPageContentV1,
    ProviderLogQueryV1,
    ProviderReadPurposeV1,
    ProviderRunObservationContentV1,
    ProviderRunPhaseV1,
    ProviderRunReadRequestV1,
    VerificationVerdictV1,
    WorkflowRecordV1,
)


@dataclass(frozen=True, slots=True)
class FakeEffectResultV1:
    disposition: ObservationDisposition
    finality_proof: object | None = None

    def __post_init__(self) -> None:
        if type(self.disposition) is not ObservationDisposition:
            raise TypeError("disposition must be exact ObservationDisposition")
        if self.disposition is not ObservationDisposition.DEFINITELY_ABSENT and self.finality_proof is not None:
            raise ValueError("only definitely-absent results may carry finality proof")


@dataclass(frozen=True, slots=True)
class FakeEffectScriptV1:
    effect_id: str
    command_digest: str
    executor_descriptor_digest: str
    kind: EffectKind
    dispatch: tuple[FakeEffectResultV1, ...]
    reconciliation: tuple[FakeEffectResultV1, ...] = ()
    stage_ref: str = "stage-output"
    provider_job_ref: str = "job-1"
    cancel_reason_digest: str | None = None

    def __post_init__(self) -> None:
        if type(self.kind) is not EffectKind:
            raise TypeError("kind must be exact EffectKind")
        if not self.dispatch or any(type(item) is not FakeEffectResultV1 for item in self.dispatch):
            raise ValueError("dispatch script must be a nonempty exact tuple")
        if type(self.reconciliation) is not tuple or any(
            type(item) is not FakeEffectResultV1 for item in self.reconciliation
        ):
            raise TypeError("reconciliation script must be an exact tuple")
        for value in (self.command_digest, self.executor_descriptor_digest):
            if type(value) is not str or len(value) != 64:
                raise ValueError("script digests must be exact SHA-256 text")
        if self.kind is EffectKind.CANCEL:
            if type(self.cancel_reason_digest) is not str or len(self.cancel_reason_digest) != 64:
                raise ValueError("cancel script requires reason digest")
        elif self.cancel_reason_digest is not None:
            raise ValueError("non-cancel script cannot carry cancellation reason")
        safe_ref(self.effect_id, "effect_id")
        safe_ref(self.stage_ref, "stage_ref")
        safe_ref(self.provider_job_ref, "provider_job_ref")


@dataclass(frozen=True, slots=True)
class FakeArtifactV1:
    descriptor: VerifiedArtifact
    content: bytes

    def __post_init__(self) -> None:
        if type(self.descriptor) is not VerifiedArtifact or type(self.content) is not bytes:
            raise TypeError("artifact descriptor/content types invalid")
        if (
            len(self.content) != self.descriptor.size_bytes
            or hashlib.sha256(self.content).hexdigest() != self.descriptor.sha256
        ):
            raise ValueError("artifact bytes do not match descriptor")


@dataclass(frozen=True, slots=True)
class FakeProviderConfigV1:
    provider: ProviderRef
    descriptor: ProviderDescriptor
    profile_digest: str
    account_ref: str
    namespace_ref: str
    executor_descriptor: ExecutorDescriptorV1
    adapter_descriptor: AdapterDescriptorV1
    effects: tuple[FakeEffectScriptV1, ...]
    run_phases: tuple[ProviderRunPhaseV1, ...]
    logs: tuple[RunLogEntry, ...]
    artifacts: tuple[FakeArtifactV1, ...]
    observed_at: str = "2026-08-27T12:00:00Z"

    def __post_init__(self) -> None:
        if type(self.provider) is not ProviderRef or type(self.descriptor) is not ProviderDescriptor:
            raise TypeError("provider identity types invalid")
        identities = (
            self.descriptor.provider_id,
            self.executor_descriptor.provider_id,
            self.adapter_descriptor.provider_id,
        )
        if identities != (self.provider.provider_id,) * 3:
            raise ValueError("fake role descriptors do not share configured provider")
        digest_text(self.profile_digest, "profile_digest")
        safe_ref(self.account_ref, "account_ref")
        safe_ref(self.namespace_ref, "namespace_ref")
        if type(self.effects) is not tuple or any(type(item) is not FakeEffectScriptV1 for item in self.effects):
            raise TypeError("effects must be an exact tuple")
        if len({item.command_digest for item in self.effects}) != len(self.effects):
            raise ValueError("effect command digests must be unique")
        if any(item.executor_descriptor_digest != self.executor_descriptor.digest for item in self.effects):
            raise ValueError("effect script executor descriptor mismatch")
        if not self.run_phases or any(type(item) is not ProviderRunPhaseV1 for item in self.run_phases):
            raise ValueError("run phases must be a nonempty exact tuple")
        if type(self.logs) is not tuple or any(type(item) is not RunLogEntry for item in self.logs):
            raise TypeError("logs must be an exact tuple")
        if any(left.sequence >= right.sequence for left, right in zip(self.logs, self.logs[1:])):
            raise ValueError("fake log sequences must be strictly increasing")
        if type(self.artifacts) is not tuple or any(type(item) is not FakeArtifactV1 for item in self.artifacts):
            raise TypeError("artifacts must be an exact tuple")
        roles = tuple(item.descriptor.role for item in self.artifacts)
        if roles != tuple(sorted(roles)) or len(roles) != len(set(roles)):
            raise ValueError("fake artifacts must have unique ascending roles")


class FakeTraceV1:
    def __init__(self) -> None:
        self._lock = Lock()
        self._events: list[tuple[str, str]] = []

    def add(self, role: str, action: str) -> None:
        with self._lock:
            self._events.append((role, action))

    def snapshot(self) -> tuple[tuple[str, str], ...]:
        with self._lock:
            return tuple(self._events)


class _ScriptCursor:
    def __init__(self) -> None:
        self._lock = Lock()
        self._positions: dict[tuple[str, str], int] = {}

    def next(self, key: tuple[str, str], values: tuple[object, ...]):
        with self._lock:
            position = self._positions.get(key, 0)
            self._positions[key] = position + 1
        return values[min(position, len(values) - 1)]


def _script(config: FakeProviderConfigV1, command_digest: str) -> FakeEffectScriptV1:
    matches = tuple(item for item in config.effects if item.command_digest == command_digest)
    if len(matches) != 1:
        raise ValueError("command is not configured exactly once")
    return matches[0]


def _observation(config, script, result, resolution_digest, epoch):
    values = {}
    if result.disposition is ObservationDisposition.FOUND:
        if script.kind is EffectKind.STAGE:
            values["stage_ref"] = ProviderStageRefV1(
                config.provider.provider_id,
                config.provider.profile_ref,
                config.account_ref,
                config.namespace_ref,
                script.stage_ref,
            )
        elif script.kind is EffectKind.SUBMIT:
            values["provider_run"] = ScopedProviderRunRefV1(
                config.provider.provider_id,
                config.provider.profile_ref,
                config.account_ref,
                config.namespace_ref,
                script.provider_job_ref,
            )
        else:
            values["cancellation"] = CancellationRefV1(
                ProviderRunRefV1(script.provider_job_ref), script.cancel_reason_digest
            )
    return ProviderObservationV1(
        script.effect_id,
        script.command_digest,
        script.executor_descriptor_digest,
        result.disposition,
        resolution_digest,
        epoch,
        finality_proof=result.finality_proof,
        **values,
    )


class FakeEffectExecutorV1:
    effect_kinds = ("stage", "submit", "cancel")
    payload_schemas = ("stage-payload/v2", "submit-payload/v2", "cancel-payload/v2")

    def __init__(self, config, scripts, trace, cursor):
        self._config, self._scripts, self._trace, self._cursor = config, scripts, trace, cursor
        self.descriptor = config.executor_descriptor
        self.provider_id = config.provider.provider_id
        self.profile_ref = config.provider.profile_ref
        self.account_ref = config.account_ref
        self.namespace_ref = config.namespace_ref

    def execute_once(self, payload, request):
        self._trace.add("executor", request.effect_kind)
        script = self._scripts[request.command_digest]
        if script.kind.value != request.effect_kind:
            raise ValueError("effect kind does not match script")
        result = self._cursor.next((script.command_digest, "dispatch"), script.dispatch)
        return _observation(self._config, script, result, request.digest, 1)


class FakeExecutorResolverV1:
    def __init__(self, executor, trace):
        self._executor, self._trace = executor, trace

    def resolve(self, request: ExecutionResolutionRequestV2):
        self._trace.add("executor_resolver", request.effect_kind)
        return mint_resolved_executor(request, self._executor)


class FakeReconciliationAdapterV1:
    capabilities = ("lookup",)

    def __init__(self, config, scripts, trace, cursor):
        self._config, self._scripts, self._trace, self._cursor = config, scripts, trace, cursor
        self.descriptor = config.adapter_descriptor
        self.provider_id = config.provider.provider_id
        self.profile_ref = config.provider.profile_ref
        self.account_ref = config.account_ref
        self.namespace_ref = config.namespace_ref

    def lookup(self, target, preparation):
        self._trace.add("adapter", "lookup")
        script = self._scripts[target.command_digest]
        if not script.reconciliation:
            raise RuntimeError("reconciliation script exhausted")
        result = self._cursor.next(
            (script.command_digest, "reconciliation"), script.reconciliation
        )
        return _observation(
            self._config, script, result, target.resolution_digest, target.ownership_epoch
        )


class FakeReconciliationResolverV1:
    def __init__(self, adapter, trace):
        self._adapter, self._trace = adapter, trace

    def resolve(self, request: ReconciliationResolutionRequestV2):
        self._trace.add("adapter_resolver", "lookup")
        return mint_resolved_adapter(request, self._adapter)


class FakeProviderEvidenceAuthorityV1:
    def __init__(self, authority_ref: str, key: bytes):
        if type(key) is not bytes or len(key) < 32:
            raise ValueError("fake evidence key must contain at least 32 bytes")
        self.authority_ref = safe_ref(authority_ref, "authority_ref")
        self.key_ref = "fake-key-v1"
        self._key = bytes(key)

    def _tag(self, domain: bytes, digest: str) -> str:
        return hmac.new(self._key, domain + digest.encode("ascii"), hashlib.sha256).hexdigest()

    def observation(self, content):
        return AuthenticatedProviderRunObservationV1(
            content,
            self.authority_ref,
            self.key_ref,
            self._tag(b"fake-observation/v1\0", content.content_digest),
        )

    def log_page(self, content):
        return AuthenticatedProviderLogPageV1(
            content,
            self.authority_ref,
            self.key_ref,
            self._tag(b"fake-log-page/v1\0", content.content_digest),
        )

    def verification(self, content):
        return AuthenticatedArtifactVerificationReceiptV1(
            content,
            self.authority_ref,
            self.key_ref,
            self._tag(b"fake-verification/v1\0", content.content_digest),
        )

    def authenticate(self, value) -> bool:
        try:
            if type(value) is AuthenticatedProviderRunObservationV1:
                expected = self._tag(b"fake-observation/v1\0", value.content.content_digest)
            elif type(value) is AuthenticatedProviderLogPageV1:
                expected = self._tag(b"fake-log-page/v1\0", value.content.content_digest)
            elif type(value) is AuthenticatedArtifactVerificationReceiptV1:
                expected = self._tag(b"fake-verification/v1\0", value.content.content_digest)
            else:
                return False
            return (
                value.authority_ref == self.authority_ref
                and value.key_ref == self.key_ref
                and hmac.compare_digest(value.tag, expected)
            )
        except Exception:
            return False


class FakeProviderRunReaderV1:
    def __init__(
        self,
        config,
        authority,
        foundation_authenticator,
        assessment_authenticator,
        trace,
        cursor,
    ):
        self._config, self._authority = config, authority
        self._foundation_authenticator = foundation_authenticator
        self._assessment_authenticator = assessment_authenticator
        self._trace, self._cursor = trace, cursor

    def _validate(self, request, purpose):
        if type(request) is not ProviderRunReadRequestV1 or request.purpose is not purpose:
            raise ValueError("read request purpose mismatch")
        ProviderRunReadRequestV1(
            request.purpose,
            request.source_workflow_record_digest,
            request.source_revision,
            request.run,
            request.provider_run,
            request.submit_command_bytes,
            request.foundation_record,
            request.assessment,
            request.foundation_binding,
            request.foundation_outcome,
            request.found_receipt_digest,
            request.canonical_bytes,
            request.request_digest,
        )
        record = request.foundation_record
        if self._foundation_authenticator.authenticate_grant(
            record.grant, record.command_bytes
        ) is not True:
            raise ValueError("Foundation grant authentication failed")
        if any(
            self._foundation_authenticator.authenticate_receipt(receipt) is not True
            for receipt in record.results
        ) or any(
            self._foundation_authenticator.authenticate_invalid_evidence(evidence)
            is not True
            for evidence in record.invalid_evidence
        ):
            raise ValueError("Foundation evidence authentication failed")
        assessment = type(request.assessment).parse(request.assessment.canonical_bytes)
        if (
            assessment != request.assessment
            or self._assessment_authenticator.authenticate(assessment) is not True
            or assessment.content.foundation_record_digest != record.record_digest
            or assessment.content.authenticated_receipt_digests
            != tuple(item.authenticated_receipt_digest for item in record.results)
        ):
            raise ValueError("Foundation assessment authentication failed")
        ref = request.provider_run.reference
        expected = (
            self._config.provider.provider_id,
            self._config.provider.profile_ref,
            self._config.account_ref,
            self._config.namespace_ref,
        )
        if (ref.provider_id, ref.profile_ref, ref.account_ref, ref.namespace_ref) != expected:
            raise ValueError("read request scope mismatch")
        return ref

    def observe(self, request):
        ref = self._validate(request, ProviderReadPurposeV1.OBSERVE)
        phase = self._cursor.next((ref.provider_job_ref, "observe"), self._config.run_phases)
        self._trace.add("reader", "observe")
        content = ProviderRunObservationContentV1(
            "synaptic-provider-run-observation-content/v1",
            request.request_digest,
            request.source_workflow_record_digest,
            request.source_revision,
            request.run,
            request.provider_run.binding_digest,
            ref.provider_id,
            ref.profile_ref,
            ref.account_ref,
            ref.namespace_ref,
            ref.provider_job_ref,
            phase,
            canonical_bytes({"phase": phase.value}),
            "provider_failed" if phase is ProviderRunPhaseV1.FAILED else None,
            "fake-reader",
            "1.0.0",
            self._config.observed_at,
        )
        return self._authority.observation(content)

    def logs(self, request, query):
        ref = self._validate(request, ProviderReadPurposeV1.LOGS)
        if type(query) is not ProviderLogQueryV1:
            raise TypeError("exact log query required")
        self._trace.add("reader", "logs")
        eligible = tuple(
            item
            for item in self._config.logs
            if query.after_sequence is None or item.sequence > query.after_sequence
        )
        selected = []
        total = 0
        for item in eligible:
            if len(selected) == query.limit or total + item.size_bytes > query.maximum_bytes:
                break
            selected.append(item)
            total += item.size_bytes
        truncated = len(selected) < len(eligible)
        content = ProviderLogPageContentV1(
            "synaptic-provider-log-page-content/v1",
            request.request_digest,
            query.log_query_digest,
            request.source_workflow_record_digest,
            request.source_revision,
            request.run,
            request.provider_run.binding_digest,
            ref.provider_id,
            ref.profile_ref,
            ref.account_ref,
            ref.namespace_ref,
            ref.provider_job_ref,
            query.after_sequence,
            tuple(selected),
            total,
            truncated,
            canonical_bytes({"entries": len(selected)}),
            "fake-reader",
            "1.0.0",
            self._config.observed_at,
        )
        return self._authority.log_page(content)

    def artifacts(self, request):
        ref = self._validate(request, ProviderReadPurposeV1.ARTIFACTS)
        self._trace.add("reader", "artifacts")
        return ArtifactManifestV1.build(
            run=request.run,
            provider_run=ref,
            artifacts=tuple(item.descriptor for item in self._config.artifacts),
            artifact_source_digest=domain_digest(
                "synaptic-fake-artifact-source/v1",
                canonical_bytes({"provider_job_ref": ref.provider_job_ref}),
            ),
            canonical_evidence=canonical_bytes(
                {"roles": [item.descriptor.role for item in self._config.artifacts]}
            ),
        )

    def iter_artifact_bytes(self, request, manifest, role, *, maximum_bytes):
        self._validate(request, ProviderReadPurposeV1.ARTIFACTS)
        matches = tuple(item for item in self._config.artifacts if item.descriptor.role == role)
        if len(matches) != 1 or matches[0].descriptor.size_bytes > maximum_bytes:
            raise ValueError("artifact role or bound invalid")
        self._trace.add("reader", "artifact_bytes")
        for offset in range(0, len(matches[0].content), 1_048_576):
            yield matches[0].content[offset : offset + 1_048_576]


class FakeProviderReaderFactoryV1:
    def __init__(self, config, reader, trace):
        self._config, self._reader, self._trace = config, reader, trace

    def create(self, request):
        self._trace.add("reader_factory", "create")
        expected = (
            self._config.provider,
            self._config.descriptor.descriptor_digest,
            self._config.profile_digest,
            self._config.account_ref,
            self._config.namespace_ref,
        )
        actual = (
            request.provider,
            request.provider_descriptor_digest,
            request.profile_digest,
            request.account_ref,
            request.namespace_ref,
        )
        if actual != expected:
            raise ValueError("reader factory request mismatch")
        return ResolvedProviderReaderV1(
            request.request_digest, *actual, self._reader
        )


class FakeArtifactVerifierV1:
    def __init__(self, authority, trace, checked_at):
        self._authority, self._trace, self._checked_at = authority, trace, checked_at

    def _receipt(self, workflow, manifest):
        content = ArtifactVerificationContentV1(
            "synaptic-artifact-verification-content/v1",
            workflow.record_digest,
            workflow.revision,
            workflow.run,
            workflow.provider_run_ref.binding_digest,
            manifest.manifest_digest,
            manifest.artifact_source_digest,
            manifest.artifacts,
            manifest.artifacts,
            VerificationVerdictV1.VERIFIED,
            None,
            "fake-verifier",
            "1.0.0",
            canonical_bytes({"verified": True}),
            self._checked_at,
        )
        return self._authority.verification(content)

    def verify(self, workflow, manifest):
        self._trace.add("verifier", "verify")
        return self._receipt(workflow, manifest)

    def replay(self, workflow, manifest, prior_receipt):
        self._trace.add("verifier", "replay")
        return self._receipt(workflow, manifest)

    def authenticate(self, receipt):
        return self._authority.authenticate(receipt)


class FakeProviderFamilyV1:
    def __init__(
        self,
        config: FakeProviderConfigV1,
        *,
        evidence_key: bytes,
        foundation_authenticator,
        assessment_authenticator,
    ):
        if type(config) is not FakeProviderConfigV1:
            raise TypeError("config must be exact FakeProviderConfigV1")
        self.config = config
        self.trace = FakeTraceV1()
        cursor = _ScriptCursor()
        self._scripts = {item.command_digest: item for item in config.effects}
        self.evidence_authority = FakeProviderEvidenceAuthorityV1(
            "fake-provider-evidence", evidence_key
        )
        self.executor = FakeEffectExecutorV1(config, self._scripts, self.trace, cursor)
        self.executor_resolver = FakeExecutorResolverV1(self.executor, self.trace)
        self.adapter = FakeReconciliationAdapterV1(config, self._scripts, self.trace, cursor)
        self.reconciliation_resolver = FakeReconciliationResolverV1(
            self.adapter, self.trace
        )
        self.reader = FakeProviderRunReaderV1(
            config,
            self.evidence_authority,
            foundation_authenticator,
            assessment_authenticator,
            self.trace,
            cursor,
        )
        self.reader_factory = FakeProviderReaderFactoryV1(
            config, self.reader, self.trace
        )
        self.artifact_verifier = FakeArtifactVerifierV1(
            self.evidence_authority, self.trace, config.observed_at
        )

    def register_command(
        self,
        command,
        *,
        dispatch: tuple[FakeEffectResultV1, ...],
        reconciliation: tuple[FakeEffectResultV1, ...] = (),
    ) -> None:
        effect = command.operation.effect
        reason = (
            command.to_dict()["cancellation"]["reason_digest"]
            if effect.kind is EffectKind.CANCEL
            else None
        )
        script = FakeEffectScriptV1(
            effect.effect_id,
            command.digest,
            command.executor.digest,
            effect.kind,
            dispatch,
            reconciliation,
            provider_job_ref=(
                effect.cancel_target.provider_job_ref
                if effect.kind is EffectKind.CANCEL
                else "job-1"
            ),
            cancel_reason_digest=reason,
        )
        retained = self._scripts.setdefault(command.digest, script)
        if retained != script:
            raise ValueError("command script conflicts with retained binding")


__all__: list[str] = []
