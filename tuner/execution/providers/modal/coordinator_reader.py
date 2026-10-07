"""Authenticated, bounded Modal reader for the generic coordinator.

This module deliberately defines no provider registration or public surface.  Its
structural command-binding and read-transport ports are private integration seams
until the Foundation-native Modal executor wire is complete.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, fields
import hashlib
from threading import Lock
from typing import Protocol

from synaptic_tuner.api.v1.results import VerifiedArtifact
from synaptic_tuner.api.v1.runs_facade import RunLogEntry

from ...coordinator_v1.model import (
    ArtifactManifestV1,
    AuthenticatedProviderLogPageV1,
    AuthenticatedProviderRunObservationV1,
    EffectIntentV1,
    FoundationDispositionV1,
    ProviderLogPageContentV1,
    ProviderLogQueryV1,
    ProviderReadPurposeV1,
    ProviderRunObservationContentV1,
    ProviderRunPhaseV1,
    ProviderRunReadRequestV1,
)
from ...foundation_v2.canonical import (
    canonical_bytes, digest_text, domain_digest, parse_canonical_object,
)
from ...foundation_v2.commands import parse_exact_command
from ...foundation_v2.identities import EffectKind
from ...coordinator_v1.state_machine import _derive_foundation
from ...coordinator_v1.ports import (
    FoundationEvidenceAuthenticatorPortV1,
    FoundationRecordAssessmentPortV1,
)
from .binding import ModalClientBinding
from .coordinator_binding import ModalCommandBinding
from .contracts import ArtifactMemberV1, BoundsPolicyV1, TerminalEvidenceV1, canonical_path
from .control import CrossPlaneIdentityV1
from .manifest import CompletionManifestV1
from .resolution import VerifiedModalDeploymentIdentityV1


class ModalCoordinatorReaderError(RuntimeError):
    """Closed non-secret failure from the coordinator Modal reader."""


class _CommandCatalog(Protocol):
    def resolve(self, command_digest: str) -> ModalCommandBinding: ...


class _BindingAuthority(Protocol):
    def authenticate(self, value: object) -> bool: ...


class _EvidenceAuthority(Protocol):
    def observation(self, content: ProviderRunObservationContentV1) -> AuthenticatedProviderRunObservationV1: ...
    def log_page(self, content: ProviderLogPageContentV1) -> AuthenticatedProviderLogPageV1: ...


def _identity_document(value: CrossPlaneIdentityV1) -> dict[str, object]:
    return {
        "account_ref": value.binding.account_ref,
        "workspace_ref": value.binding.workspace_ref,
        "environment_ref": value.binding.environment_ref,
        "client_ref": value.binding.client_ref,
        "sdk_version": value.binding.sdk_version,
        "control_volume_id": value.control_volume_id,
        "artifact_volume_id": value.artifact_volume_id,
        "job_ref": value.job_ref,
        "effect_id": value.effect_id,
        "command_digest": value.command_digest,
        "plan_digest": value.plan_digest,
        "deployment_attestation_digest": value.deployment_attestation_digest,
        "invocation_nonce": value.invocation_nonce,
        "generation": value.generation,
        "key_ref": value.key_ref,
    }


@dataclass(frozen=True, slots=True)
class ModalTerminalSnapshot:
    """Transport-authenticated observation retaining exact cross-plane identity."""

    phase: ProviderRunPhaseV1
    identity: CrossPlaneIdentityV1
    canonical_evidence: bytes
    terminal: TerminalEvidenceV1 | None

    @classmethod
    def build(cls, phase, identity, terminal=None):
        evidence = canonical_bytes({
            "phase": phase.value, "identity": _identity_document(identity),
            "terminal": None if terminal is None else {
                field.name: getattr(terminal, field.name)
                for field in fields(TerminalEvidenceV1)
            },
        })
        return cls(phase, identity, evidence, terminal)

    @property
    def expected_evidence(self) -> bytes:
        return canonical_bytes({
            "phase": self.phase.value,
            "identity": _identity_document(self.identity),
            "terminal": None if self.terminal is None else {
                field.name: getattr(self.terminal, field.name)
                for field in fields(TerminalEvidenceV1)
            },
        })

    def __post_init__(self) -> None:
        if type(self.phase) is not ProviderRunPhaseV1 or type(self.identity) is not CrossPlaneIdentityV1:
            raise TypeError("exact Modal terminal snapshot values required")
        if (type(self.canonical_evidence) is not bytes or not self.canonical_evidence
                or len(self.canonical_evidence) > 65536):
            raise ValueError("terminal evidence must be bounded nonempty bytes")
        if self.canonical_evidence != self.expected_evidence:
            raise ValueError("terminal evidence does not bind snapshot")
        terminal_phase = self.phase in {
            ProviderRunPhaseV1.SUCCEEDED, ProviderRunPhaseV1.FAILED,
            ProviderRunPhaseV1.CANCELLED,
        }
        if terminal_phase != (type(self.terminal) is TerminalEvidenceV1):
            raise ValueError("terminal state/evidence matrix invalid")
        if self.terminal is not None:
            expected = {
                ProviderRunPhaseV1.SUCCEEDED: "completed",
                ProviderRunPhaseV1.FAILED: "failed",
                ProviderRunPhaseV1.CANCELLED: "cancelled",
            }[self.phase]
            if self.terminal.status_code != expected or not self.identity.matches_record(self.terminal):
                raise ValueError("terminal evidence identity mismatch")


@dataclass(frozen=True, slots=True)
class ModalLogSnapshot:
    """Authenticated/relisted Modal log snapshot with terminal watermark."""

    entries: tuple[RunLogEntry, ...]
    identity: CrossPlaneIdentityV1
    chain_digest: str
    query_digest: str
    generation: int
    high_watermark_sequence: int
    terminal_phase: ProviderRunPhaseV1 | None
    terminal_generation: int | None
    truncated: bool
    canonical_evidence: bytes

    @classmethod
    def build(cls, entries, identity, chain_digest, query_digest, generation,
              high_watermark_sequence, terminal_phase=None, terminal_generation=None,
              truncated=False):
        evidence = canonical_bytes({
            "identity": _identity_document(identity), "chain_digest": chain_digest,
            "query_digest": query_digest, "generation": generation,
            "high_watermark_sequence": high_watermark_sequence,
            "terminal_phase": None if terminal_phase is None else terminal_phase.value,
            "terminal_generation": terminal_generation,
            "truncated": truncated,
            "entries": [item.to_dict() for item in entries],
        })
        return cls(tuple(entries), identity, chain_digest, query_digest, generation,
                   high_watermark_sequence, terminal_phase, terminal_generation,
                   truncated, evidence)

    @property
    def expected_evidence(self) -> bytes:
        return canonical_bytes({
            "identity": _identity_document(self.identity),
            "chain_digest": self.chain_digest,
            "query_digest": self.query_digest,
            "generation": self.generation,
            "high_watermark_sequence": self.high_watermark_sequence,
            "terminal_phase": None if self.terminal_phase is None else self.terminal_phase.value,
            "terminal_generation": self.terminal_generation,
            "truncated": self.truncated,
            "entries": [item.to_dict() for item in self.entries],
        })

    def __post_init__(self) -> None:
        if (type(self.entries) is not tuple or any(type(x) is not RunLogEntry for x in self.entries)
                or type(self.identity) is not CrossPlaneIdentityV1):
            raise TypeError("exact Modal log snapshot values required")
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("invalid log generation")
        digest_text(self.chain_digest, "chain_digest")
        digest_text(self.query_digest, "query_digest")
        if type(self.high_watermark_sequence) is not int or self.high_watermark_sequence < -1:
            raise ValueError("invalid log high watermark")
        sequences = tuple(item.sequence for item in self.entries)
        if any(a >= b for a, b in zip(sequences, sequences[1:])):
            raise ValueError("log sequence is not strictly increasing")
        if sequences and sequences[-1] > self.high_watermark_sequence:
            raise ValueError("log entry exceeds high watermark")
        if (self.terminal_phase is None) != (self.terminal_generation is None):
            raise ValueError("log terminal watermark is incomplete")
        if self.terminal_phase is not None and self.terminal_phase not in {
            ProviderRunPhaseV1.SUCCEEDED, ProviderRunPhaseV1.FAILED,
            ProviderRunPhaseV1.CANCELLED,
        }:
            raise ValueError("log terminal phase is not closed")
        if self.terminal_generation is not None and self.terminal_generation != self.generation:
            raise ValueError("log terminal generation mismatch")
        if type(self.truncated) is not bool:
            raise TypeError("log truncation flag must be exact")
        if (type(self.canonical_evidence) is not bytes or not self.canonical_evidence
                or len(self.canonical_evidence) > 65536
                or self.canonical_evidence != self.expected_evidence):
            raise ValueError("log evidence must be bounded nonempty bytes")


@dataclass(frozen=True, slots=True)
class ModalArtifactInventory:
    """Authenticated completion inventory; artifact bodies are not read here."""

    manifest: CompletionManifestV1
    identity: CrossPlaneIdentityV1
    canonical_evidence: bytes

    @classmethod
    def build(cls, manifest, identity):
        evidence = canonical_bytes({
            "identity": _identity_document(identity),
            "terminal_evidence_digest": manifest.terminal_evidence_digest,
            "log_chain_digest": manifest.log_chain_digest,
            "artifact_set_digest": manifest.artifact_set_digest,
            "members": [{
                "role": item.role.value, "path": item.path, "size": item.size,
                "sha256": item.sha256, "provider_entry_id": item.provider_entry_id,
            } for item in sorted(manifest.members, key=lambda member: member.role.value)],
        })
        return cls(manifest, identity, evidence)

    @property
    def expected_evidence(self) -> bytes:
        value = self.manifest
        return canonical_bytes({
            "identity": _identity_document(self.identity),
            "terminal_evidence_digest": value.terminal_evidence_digest,
            "log_chain_digest": value.log_chain_digest,
            "artifact_set_digest": value.artifact_set_digest,
            "members": [{
                "role": item.role.value, "path": item.path, "size": item.size,
                "sha256": item.sha256, "provider_entry_id": item.provider_entry_id,
            } for item in sorted(value.members, key=lambda member: member.role.value)],
        })

    def __post_init__(self) -> None:
        if type(self.manifest) is not CompletionManifestV1 or type(self.identity) is not CrossPlaneIdentityV1:
            raise TypeError("exact Modal inventory values required")
        if not self.identity.matches_record(self.manifest):
            raise ValueError("completion inventory identity mismatch")
        if (type(self.canonical_evidence) is not bytes or not self.canonical_evidence
                or len(self.canonical_evidence) > 65536
                or self.canonical_evidence != self.expected_evidence):
            raise ValueError("inventory evidence must be bounded nonempty bytes")


class _AuthenticatedReadTransport(Protocol):
    def observe(self, binding: ModalCommandBinding, *, provider_job_ref: str) -> ModalTerminalSnapshot: ...
    def logs(self, binding: ModalCommandBinding, query: ProviderLogQueryV1, *, provider_job_ref: str) -> ModalLogSnapshot: ...
    def artifact_inventory(self, binding: ModalCommandBinding, *, provider_job_ref: str) -> ModalArtifactInventory: ...
    def iter_artifact(self, binding: ModalCommandBinding, member: ArtifactMemberV1, *, provider_job_ref: str, maximum_bytes: int) -> Iterator[bytes]: ...


def _rebuild(value, expected_type):
    if type(value) is not expected_type:
        raise ValueError
    rebuilt = expected_type(**{field.name: getattr(value, field.name) for field in fields(expected_type)})
    if rebuilt != value:
        raise ValueError
    return rebuilt


class ModalCoordinatorRunReader:
    """Foundation-authenticated Modal implementation of ProviderRunReaderPortV1."""

    def __init__(self, *, catalog: _CommandCatalog, binding_authority: _BindingAuthority,
                 foundation_authenticator: FoundationEvidenceAuthenticatorPortV1,
                 assessment_authenticator: FoundationRecordAssessmentPortV1,
                 evidence_authority: _EvidenceAuthority,
                 transport: _AuthenticatedReadTransport, observed_at: str,
                 bounds: BoundsPolicyV1 = BoundsPolicyV1()) -> None:
        for value, method in ((catalog, "resolve"), (binding_authority, "authenticate"),
                              (assessment_authenticator, "authenticate"),
                              (evidence_authority, "observation"), (transport, "observe")):
            if not callable(getattr(value, method, None)):
                raise TypeError("Modal reader dependency is incomplete")
        self._catalog = catalog
        self._binding_authority = binding_authority
        self._foundation = foundation_authenticator
        self._assessments = assessment_authenticator
        self._authority = evidence_authority
        self._transport = transport
        self._observed_at = observed_at
        self._bounds = bounds
        self._log_lock = Lock()
        self._log_states = {}
        self._log_commitments = {}

    def _validate(self, request, purpose):
        try:
            if type(request) is not ProviderRunReadRequestV1 or request.purpose is not purpose:
                raise ValueError
            _rebuild(request, ProviderRunReadRequestV1)
            record = request.foundation_record
            intent = EffectIntentV1.from_command_bytes(request.submit_command_bytes)
            command = parse_exact_command(request.submit_command_bytes)
            if command.operation.effect.kind is not EffectKind.SUBMIT:
                raise ValueError
            derived_binding, derived_outcome, derived_run = _derive_foundation(
                intent, record, request.assessment, self._foundation,
                self._assessments, None,
            )
            if (derived_binding, derived_outcome, derived_run) != (
                    request.foundation_binding, request.foundation_outcome,
                    request.provider_run) or derived_run is None:
                raise ValueError
            if (derived_outcome.disposition is not FoundationDispositionV1.FOUND
                    or request.found_receipt_digest
                    != derived_run.authenticated_receipt_digest):
                raise ValueError
            prep = command.preparation
            ref = derived_run.reference
            if ((request.run.project_ref, request.run.run_id)
                    != (prep.project_ref, prep.run_id)
                    or (ref.provider_id, ref.profile_ref, ref.account_ref, ref.namespace_ref)
                    != (prep.provider.provider_id, prep.provider.profile_ref,
                        prep.scope.account_ref, prep.scope.namespace_ref)):
                raise ValueError

            retained = self._catalog.resolve(command.digest)
            if type(retained) is not ModalCommandBinding:
                raise ValueError
            binding = ModalCommandBinding(
                retained.command_bytes, retained.preparation_snapshot,
                retained.deployment_bytes,
            )
            if binding != retained or self._binding_authority.authenticate(binding) is not True:
                raise ValueError
            if (binding.command_bytes != request.submit_command_bytes
                    or parse_exact_command(binding.command_bytes).canonical_bytes
                    != command.canonical_bytes
                    or (binding.command_digest, binding.provider_id, binding.profile_ref,
                        binding.account_ref, binding.namespace_ref)
                    != (command.digest, prep.provider.provider_id, prep.provider.profile_ref,
                        prep.scope.account_ref, prep.scope.namespace_ref)
                    or type(binding.client_binding) is not ModalClientBinding
                    or type(binding.deployment) is not VerifiedModalDeploymentIdentityV1):
                raise ValueError
            selection = binding.deployment.selection
            if (selection.account_ref, selection.workspace_ref,
                    selection.environment_ref, selection.client_ref, selection.sdk_version) != (
                    binding.client_binding.account_ref, binding.client_binding.workspace_ref,
                    binding.client_binding.environment_ref, binding.client_binding.client_ref,
                    binding.client_binding.sdk_version):
                raise ValueError
            return binding, ref
        except Exception:
            raise ModalCoordinatorReaderError("modal_read_authentication_failed") from None

    @staticmethod
    def _identity_matches(binding, ref, identity):
        command = parse_exact_command(binding.command_bytes)
        selection = binding.deployment.selection
        return (type(identity) is CrossPlaneIdentityV1
                and identity.binding == binding.client_binding
                and (selection.account_ref, selection.workspace_ref,
                     selection.environment_ref, selection.client_ref, selection.sdk_version)
                == (binding.client_binding.account_ref, binding.client_binding.workspace_ref,
                    binding.client_binding.environment_ref, binding.client_binding.client_ref,
                    binding.client_binding.sdk_version)
                and (identity.effect_id, identity.command_digest, identity.plan_digest,
                     identity.job_ref, identity.binding.account_ref,
                     identity.deployment_attestation_digest, identity.invocation_nonce)
                == (command.operation.effect.effect_id, binding.command_digest,
                    command.preparation.plan_fingerprint, ref.provider_job_ref,
                    ref.account_ref, binding.deployment.attestation_digest,
                    command.operation.invocation_nonce))

    def _accept_log_snapshot(self, snapshot, query):
        if snapshot.query_digest != query.log_query_digest:
            raise ValueError
        stream = (snapshot.identity.effect_id, snapshot.identity.job_ref)
        current = (
            snapshot.generation, snapshot.high_watermark_sequence,
            snapshot.terminal_phase, snapshot.terminal_generation,
        )
        commitment_key = (stream, snapshot.generation, snapshot.query_digest)
        commitment = hashlib.sha256(snapshot.canonical_evidence).hexdigest()
        with self._log_lock:
            previous = self._log_states.get(stream)
            if previous is not None:
                prior_generation, prior_high, prior_phase, prior_terminal_generation = previous
                if snapshot.generation < prior_generation or snapshot.high_watermark_sequence < prior_high:
                    raise ValueError
                if prior_phase is not None and current != (
                        prior_terminal_generation, prior_high, prior_phase,
                        prior_terminal_generation):
                    raise ValueError
            retained = self._log_commitments.get(commitment_key)
            if retained is not None and retained != commitment:
                raise ValueError
            if retained is None and len(self._log_commitments) >= 1024:
                raise ValueError
            self._log_states[stream] = current
            self._log_commitments[commitment_key] = commitment

    def observe(self, request):
        binding, ref = self._validate(request, ProviderReadPurposeV1.OBSERVE)
        try:
            snapshot = self._transport.observe(binding, provider_job_ref=ref.provider_job_ref)
            if type(snapshot) is not ModalTerminalSnapshot or not self._identity_matches(binding, ref, snapshot.identity):
                raise ValueError
            diagnostic = "modal_run_failed" if snapshot.phase is ProviderRunPhaseV1.FAILED else None
            content = ProviderRunObservationContentV1(
                "synaptic-provider-run-observation-content/v1", request.request_digest,
                request.source_workflow_record_digest, request.source_revision, request.run,
                request.provider_run.binding_digest, ref.provider_id, ref.profile_ref,
                ref.account_ref, ref.namespace_ref, ref.provider_job_ref, snapshot.phase,
                snapshot.canonical_evidence, diagnostic, "modal-coordinator-reader", "0.1.0",
                self._observed_at,
            )
            owned = self._authority.observation(content)
            if type(owned) is not AuthenticatedProviderRunObservationV1 or owned.content != content:
                raise ValueError
            return owned
        except ModalCoordinatorReaderError:
            raise
        except Exception:
            raise ModalCoordinatorReaderError("modal_read_evidence_invalid") from None

    def logs(self, request, query):
        binding, ref = self._validate(request, ProviderReadPurposeV1.LOGS)
        try:
            if type(query) is not ProviderLogQueryV1:
                raise ValueError
            snapshot = self._transport.logs(binding, query, provider_job_ref=ref.provider_job_ref)
            if type(snapshot) is not ModalLogSnapshot or not self._identity_matches(binding, ref, snapshot.identity):
                raise ValueError
            selected = snapshot.entries
            total = sum(item.size_bytes for item in selected)
            if (len(selected) > query.limit or total > query.maximum_bytes
                    or any(b != a + 1 for a, b in zip(
                        (item.sequence for item in selected),
                        (item.sequence for item in selected[1:])))
                    or (query.after_sequence is not None and selected
                        and selected[0].sequence != query.after_sequence + 1)):
                raise ValueError
            last_seen = selected[-1].sequence if selected else (
                query.after_sequence if query.after_sequence is not None else -1
            )
            if snapshot.truncated != (last_seen < snapshot.high_watermark_sequence):
                raise ValueError
            content = ProviderLogPageContentV1(
                "synaptic-provider-log-page-content/v1", request.request_digest,
                query.log_query_digest, request.source_workflow_record_digest,
                request.source_revision, request.run, request.provider_run.binding_digest,
                ref.provider_id, ref.profile_ref, ref.account_ref, ref.namespace_ref,
                ref.provider_job_ref, query.after_sequence, tuple(selected), total,
                snapshot.truncated,
                snapshot.canonical_evidence, "modal-coordinator-reader", "0.1.0",
                self._observed_at,
            )
            owned = self._authority.log_page(content)
            if type(owned) is not AuthenticatedProviderLogPageV1 or owned.content != content:
                raise ValueError
            # Retain monotone state only after the entire provider page and the
            # authority-produced envelope have passed validation.  The lock in
            # _accept_log_snapshot keeps compare-and-commit atomic.
            self._accept_log_snapshot(snapshot, query)
            return owned
        except Exception:
            raise ModalCoordinatorReaderError("modal_read_evidence_invalid") from None

    def native_artifacts(self, request):
        """Authenticate native placement and its generic manifest together.

        This provider-internal projection retains the exact command binding,
        provider reference, native inventory and generic manifest for consumers
        that prepare artifacts on the provider. It reads no artifact bodies and
        grants no serving or mutation authority. Callers must bind the result to
        their current retained workflow, not just compare public artifact hashes.
        """
        binding, ref = self._validate(request, ProviderReadPurposeV1.ARTIFACTS)
        try:
            inventory = self._transport.artifact_inventory(
                binding, provider_job_ref=ref.provider_job_ref
            )
            if type(inventory) is not ModalArtifactInventory or not self._identity_matches(
                    binding, ref, inventory.identity):
                raise ValueError
            members = tuple(sorted(inventory.manifest.members, key=lambda item: item.role.value))
            expected_prefix = f"operations/{request.provider_run.effect_id}/output/"
            if any(not item.path.startswith(expected_prefix) for item in members):
                raise ValueError
            artifacts = tuple(VerifiedArtifact(item.role.value, item.sha256, item.size)
                              for item in members)
            if any(item.size > self._bounds.max_artifact_bytes for item in members) or sum(
                    item.size for item in members) > self._bounds.max_artifact_total_bytes:
                raise ValueError
            evidence = canonical_bytes({
                "completion_artifact_set_digest": inventory.manifest.artifact_set_digest,
                "members": [{"role": item.role.value, "path": item.path,
                             "size": item.size, "sha256": item.sha256,
                             "provider_entry_id": item.provider_entry_id} for item in members],
            })
            manifest = ArtifactManifestV1.build(
                run=request.run, provider_run=ref, artifacts=artifacts,
                artifact_source_digest=domain_digest(
                    "synaptic-modal-artifact-source/v1", inventory.canonical_evidence
                ), canonical_evidence=evidence,
            )
            return binding, ref, inventory, manifest
        except Exception:
            raise ModalCoordinatorReaderError("modal_read_evidence_invalid") from None

    def artifacts(self, request):
        return self.native_artifacts(request)[3]

    def iter_artifact_bytes(self, request, manifest, role, *, maximum_bytes):
        binding, ref, inventory, expected_manifest = self.native_artifacts(request)
        try:
            if type(manifest) is not ArtifactManifestV1 or manifest != expected_manifest:
                raise ValueError
            matches = tuple(item for item in inventory.manifest.members if item.role.value == role)
            if (len(matches) != 1 or type(maximum_bytes) is not int or maximum_bytes < 1
                    or matches[0].size > maximum_bytes
                    or maximum_bytes > self._bounds.max_artifact_bytes):
                raise ValueError
            member = matches[0]
            canonical_path(member.path)
        except Exception:
            raise ModalCoordinatorReaderError("modal_read_bounds_invalid") from None

        def checked() -> Iterator[bytes]:
            total = 0
            digest = hashlib.sha256()
            try:
                for chunk in self._transport.iter_artifact(
                        binding, member, provider_job_ref=ref.provider_job_ref,
                        maximum_bytes=maximum_bytes):
                    if type(chunk) is not bytes or not chunk:
                        raise ValueError
                    total += len(chunk)
                    if total > maximum_bytes or total > member.size:
                        raise ValueError
                    digest.update(chunk)
                    yield chunk
                if total != member.size or digest.hexdigest() != member.sha256:
                    raise ValueError
            except Exception:
                raise ModalCoordinatorReaderError("modal_read_evidence_invalid") from None

        return checked()


__all__: list[str] = []
