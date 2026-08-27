"""Authenticated, bounded Docker observation/log/artifact reader."""

from __future__ import annotations

import hashlib
from dataclasses import fields
from threading import Lock
from synaptic_tuner.api.v1.results import VerifiedArtifact

from synaptic_tuner.api.v1.runs_facade import RunLogEntry

from ...coordinator_v1.model import (
    ArtifactManifestV1,
    AuthenticatedProviderLogPageV1,
    AuthenticatedProviderRunObservationV1,
    ProviderLogPageContentV1,
    ProviderLogQueryV1,
    ProviderReadPurposeV1,
    ProviderRunObservationContentV1,
    ProviderRunPhaseV1,
    ProviderRunReadRequestV1,
)
from ...foundation_v2.canonical import canonical_bytes, domain_digest
from ...foundation_v2.commands import parse_exact_command
from ...foundation_v2.registry import ProviderReaderFactoryRequestV1, ResolvedProviderReaderV1
from .model import (
    AuthenticatedDockerArtifactInventoryV1,
    AuthenticatedDockerLogPageV1,
    DockerArtifactChunkV1,
    DockerArtifactEOFV1,
    DockerArtifactEntryV1,
    DockerArtifactInventoryRequestV1,
    DockerArtifactReadRequestV1,
    DockerArtifactInventoryV1,
    DockerCommandBindingV1,
    DockerDiagnosticCodeV1,
    DockerLookupDispositionV1,
    DockerLookupPurposeV1,
    DockerLookupRequestV1,
    DockerLookupResultV1,
    DockerProviderError,
    DockerRunPhaseV1,
    DockerLogTerminalPhaseV1,
    DockerLogPageV1,
    DockerLogReadRequestV1,
    MAX_LOG_BYTES,
    MAX_LOG_ENTRIES,
    labels_for,
)
from .effects import _resolve_authenticated_binding


_PHASES = {
    DockerRunPhaseV1.CREATED: ProviderRunPhaseV1.QUEUED,
    DockerRunPhaseV1.RUNNING: ProviderRunPhaseV1.RUNNING,
    DockerRunPhaseV1.SUCCEEDED: ProviderRunPhaseV1.SUCCEEDED,
    DockerRunPhaseV1.FAILED: ProviderRunPhaseV1.FAILED,
    DockerRunPhaseV1.CANCELLED: ProviderRunPhaseV1.CANCELLED,
}

MAX_LOG_PAGE_COMMITMENTS = 1024


def _rebuilt(value, expected_type):
    if type(value) is not expected_type:
        raise ValueError
    rebuilt = expected_type(**{field.name: getattr(value, field.name) for field in fields(expected_type)})
    if rebuilt != value:
        raise ValueError
    return rebuilt


class DockerProviderRunReaderV1:
    def __init__(self, profile, command_catalog, binding_authority,
                 authorization, evidence_authority,
                 reader, *, observed_at: str):
        self._profile, self._catalog = profile, command_catalog
        self._binding_authority = binding_authority
        self._authorization, self._authority, self._reader = authorization, evidence_authority, reader
        self._observed_at = observed_at
        self._log_state_lock = Lock()
        self._log_states = {}
        self._log_page_commitments = {}

    def _accept_log_snapshot(self, page):
        current = (
            page.generation, page.high_watermark_sequence,
            page.terminal_phase, page.terminal_generation,
        )
        commitment_key = (
            page.log_stream_digest, page.generation, page.query_digest,
        )
        commitment = page.content_digest
        with self._log_state_lock:
            previous = self._log_states.get(page.log_stream_digest)
            if previous is not None:
                prior_generation, prior_high, prior_phase, prior_terminal_generation = previous
                if page.generation < prior_generation:
                    raise ValueError
                if page.high_watermark_sequence < prior_high:
                    raise ValueError
                if prior_phase is not None and (
                    page.terminal_phase, page.terminal_generation,
                    page.high_watermark_sequence, page.generation,
                ) != (
                    prior_phase, prior_terminal_generation,
                    prior_high, prior_terminal_generation,
                ):
                    raise ValueError
            retained_commitment = self._log_page_commitments.get(commitment_key)
            if retained_commitment is not None and retained_commitment != commitment:
                raise ValueError
            if (retained_commitment is None
                    and len(self._log_page_commitments) >= MAX_LOG_PAGE_COMMITMENTS):
                raise ValueError
            self._log_states[page.log_stream_digest] = current
            self._log_page_commitments[commitment_key] = commitment

    def _validate(self, request, purpose):
        try:
            if (type(request) is not ProviderRunReadRequestV1 or request.purpose is not purpose
                    or self._authorization.authenticate(request) is not True):
                raise ValueError
            command = parse_exact_command(request.submit_command_bytes)
            binding, bound_command = _resolve_authenticated_binding(
                self._catalog, self._binding_authority, self._profile, command.digest,
                expected_command_bytes=request.submit_command_bytes,
            )
            if bound_command.canonical_bytes != command.canonical_bytes:
                raise ValueError
            ref = request.provider_run.reference
            p = binding.plan.profile
            prep = command.preparation
            if (type(binding) is not DockerCommandBindingV1 or binding.effect_kind != "submit"
                    or command.operation.effect.effect_id != binding.effect_id
                    or command.digest != binding.command_digest
                    or (prep.preparation_digest, prep.plan_fingerprint, prep.source_digest,
                        prep.workload_digest, prep.runtime_digest, prep.resource_digest,
                        prep.artifact_contract_digest, prep.quote_digest,
                        prep.secret_requirements_digest) != (
                        binding.plan.preparation_digest, binding.plan.plan_fingerprint,
                        binding.plan.source_digest, p.workload.workload_digest,
                        p.runtime.digest, p.resource_digest, p.artifacts.digest,
                        p.quote_digest, p.secret_requirements_digest)
                    or request.run.project_ref != binding.plan.project_ref
                    or request.run.run_id != binding.plan.run_id
                    or (ref.provider_id, ref.profile_ref, ref.account_ref, ref.namespace_ref)
                    != (p.provider.provider_id, p.provider.profile_ref,
                        p.scope.account_ref, p.scope.namespace_ref)):
                raise ValueError
            return binding, ref, labels_for(binding.identity)
        except Exception:
            raise DockerProviderError(DockerDiagnosticCodeV1.AUTHENTICATION_FAILED) from None

    def observe(self, request):
        binding, ref, labels = self._validate(request, ProviderReadPurposeV1.OBSERVE)
        try:
            lookup_request = DockerLookupRequestV1(
                labels, DockerLookupPurposeV1.OBSERVE, request.source_revision
            )
            result = self._reader.lookup(lookup_request)
            if (type(result) is not DockerLookupResultV1
                    or result.disposition is not DockerLookupDispositionV1.FOUND
                    or result.labels != labels or result.container_ref != ref.provider_job_ref):
                raise ValueError
            phase = _PHASES[result.phase]
            diagnostic = "docker_run_failed" if phase is ProviderRunPhaseV1.FAILED else None
            content = ProviderRunObservationContentV1(
                "synaptic-provider-run-observation-content/v1", request.request_digest,
                request.source_workflow_record_digest, request.source_revision, request.run,
                request.provider_run.binding_digest, ref.provider_id, ref.profile_ref,
                ref.account_ref, ref.namespace_ref, ref.provider_job_ref, phase,
                canonical_bytes({"labels_digest": labels.digest, "phase": result.phase.value}),
                diagnostic, "docker-reader-v1", "1.0.0", self._observed_at,
            )
            owned = self._authority.observation(content)
            if type(owned) is not AuthenticatedProviderRunObservationV1 or owned.content != content:
                raise ValueError
            return owned
        except DockerProviderError:
            raise
        except Exception:
            raise DockerProviderError(DockerDiagnosticCodeV1.MALFORMED_EVIDENCE) from None

    def logs(self, request, query):
        _, ref, labels = self._validate(request, ProviderReadPurposeV1.LOGS)
        try:
            if type(query) is not ProviderLogQueryV1:
                raise ValueError
            docker_request = DockerLogReadRequestV1(
                labels, request.request_digest, query.log_query_digest,
                query.after_sequence, query.limit, query.maximum_bytes,
                4096, request.source_revision,
            )
            owned_page = self._reader.logs(docker_request)
            if (type(owned_page) is not AuthenticatedDockerLogPageV1
                    or self._authority.authenticate_log_page(owned_page) is not True):
                raise ValueError
            page = _rebuilt(owned_page.content, DockerLogPageV1)
            expected_stream_digest = domain_digest(
                "synaptic-docker-log-stream/v1",
                canonical_bytes({"labels_digest": labels.digest}),
            )
            if (page.request_digest, page.labels_digest, page.query_digest,
                page.log_stream_digest, page.generation, page.after_sequence, page.requested_limit,
                page.requested_maximum_bytes, page.maximum_entry_bytes) != (
                    docker_request.digest, labels.digest, query.log_query_digest,
                    expected_stream_digest,
                    request.source_revision, query.after_sequence, query.limit,
                    query.maximum_bytes, 4096):
                raise ValueError
            self._accept_log_snapshot(page)
            raw = page.entries
            if len(raw) > query.limit or sum(v.size_bytes for v in raw) > query.maximum_bytes:
                raise ValueError
            sequences = tuple(v.sequence for v in raw)
            if any(a >= b for a, b in zip(sequences, sequences[1:])):
                raise ValueError
            selected = list(raw)
            total = sum(entry.size_bytes for entry in selected)
            if total > MAX_LOG_BYTES:
                raise ValueError
            content = ProviderLogPageContentV1(
                "synaptic-provider-log-page-content/v1", request.request_digest,
                query.log_query_digest, request.source_workflow_record_digest,
                request.source_revision, request.run, request.provider_run.binding_digest,
                ref.provider_id, ref.profile_ref, ref.account_ref, ref.namespace_ref,
                ref.provider_job_ref, query.after_sequence, tuple(selected), total,
                page.truncated,
                canonical_bytes({"labels_digest": labels.digest, "last_sequence":
                    (selected[-1].sequence if selected else query.after_sequence),
                    "high_watermark_sequence": page.high_watermark_sequence,
                    "log_stream_digest": page.log_stream_digest,
                    "terminal_phase": (
                        None if page.terminal_phase is None else page.terminal_phase.value
                    ),
                    "terminal_generation": page.terminal_generation,
                    "docker_log_page_digest": page.content_digest}),
                "docker-reader-v1", "1.0.0", self._observed_at,
            )
            owned = self._authority.log_page(content)
            if type(owned) is not AuthenticatedProviderLogPageV1 or owned.content != content:
                raise ValueError
            return owned
        except Exception:
            raise DockerProviderError(DockerDiagnosticCodeV1.MALFORMED_EVIDENCE) from None

    def _inventory(self, request, purpose):
        binding, ref, labels = self._validate(request, purpose)
        try:
            plan = binding.plan
            inventory_request = DockerArtifactInventoryRequestV1(
                labels, request.request_digest, request.source_revision,
                plan.profile.profile_digest, plan.digest, plan.profile.artifacts.digest,
                plan.profile.roots.artifact_ref,
            )
            owned_inventory = self._reader.artifact_inventory(inventory_request)
            if (type(owned_inventory) is not AuthenticatedDockerArtifactInventoryV1
                    or self._authority.authenticate_inventory(owned_inventory) is not True):
                raise ValueError
            raw_inventory = owned_inventory.content
            if type(raw_inventory) is not DockerArtifactInventoryV1:
                raise ValueError
            rebuilt_entries = tuple(
                DockerArtifactEntryV1(
                    VerifiedArtifact(value.descriptor.role, value.descriptor.sha256,
                                     value.descriptor.size_bytes),
                    value.relative_path, value.file_identity_digest,
                ) for value in raw_inventory.entries
            )
            inventory = DockerArtifactInventoryV1(
                raw_inventory.labels, raw_inventory.request_digest,
                raw_inventory.generation, raw_inventory.profile_digest,
                raw_inventory.prepared_plan_digest,
                raw_inventory.artifact_contract_digest,
                raw_inventory.artifact_root_ref, rebuilt_entries,
                raw_inventory.evidence_digest,
            )
            if inventory != raw_inventory:
                raise ValueError
            if (type(inventory) is not DockerArtifactInventoryV1
                    or (inventory.labels, inventory.request_digest, inventory.generation,
                        inventory.profile_digest, inventory.prepared_plan_digest,
                        inventory.artifact_contract_digest, inventory.artifact_root_ref)
                    != (labels, inventory_request.digest, request.source_revision,
                        plan.profile.profile_digest, plan.digest,
                        plan.profile.artifacts.digest, plan.profile.roots.artifact_ref)):
                raise ValueError
            roles = tuple(v.descriptor.role for v in inventory.entries)
            contract = binding.plan.profile.artifacts
            if roles != contract.roles or any(v.descriptor.size_bytes > contract.maximum_artifact_bytes for v in inventory.entries):
                raise ValueError
            if sum(v.descriptor.size_bytes for v in inventory.entries) > contract.maximum_total_bytes:
                raise ValueError
            return binding, ref, labels, owned_inventory
        except Exception:
            raise DockerProviderError(DockerDiagnosticCodeV1.MALFORMED_EVIDENCE) from None

    def artifacts(self, request):
        try:
            _, ref, labels, owned_inventory = self._inventory(request, ProviderReadPurposeV1.ARTIFACTS)
            return self._manifest(request, ref, labels, owned_inventory)
        except DockerProviderError:
            raise
        except Exception:
            raise DockerProviderError(DockerDiagnosticCodeV1.MALFORMED_EVIDENCE) from None

    @staticmethod
    def _manifest(request, ref, labels, owned_inventory):
        inventory = owned_inventory.content
        return ArtifactManifestV1.build(
            run=request.run, provider_run=ref,
            artifacts=tuple(value.descriptor for value in inventory.entries),
            artifact_source_digest=domain_digest(
                "synaptic-docker-artifact-source/v1",
                canonical_bytes({"labels_digest": labels.digest,
                                 "profile_digest": inventory.profile_digest,
                                 "prepared_plan_digest": inventory.prepared_plan_digest,
                                 "artifact_contract_digest": inventory.artifact_contract_digest,
                                 "artifact_root_ref": inventory.artifact_root_ref,
                                 "inventory_evidence_digest": inventory.evidence_digest}),
            ),
            canonical_evidence=canonical_bytes({
                "inventory_evidence_digest": inventory.evidence_digest,
                "profile_digest": inventory.profile_digest,
                "prepared_plan_digest": inventory.prepared_plan_digest,
                "artifact_contract_digest": inventory.artifact_contract_digest,
                "artifact_root_ref": inventory.artifact_root_ref,
                "paths": [{"path": value.relative_path,
                           "file_identity_digest": value.file_identity_digest}
                          for value in inventory.entries],
            }),
        )

    def iter_artifact_bytes(self, request, manifest, role, *, maximum_bytes):
        _, ref, labels, owned_inventory = self._inventory(request, ProviderReadPurposeV1.ARTIFACTS)
        inventory = owned_inventory.content
        try:
            if type(manifest) is not ArtifactManifestV1 or manifest != self._manifest(request, ref, labels, owned_inventory):
                raise ValueError
            matches = tuple(value for value in inventory.entries if value.descriptor.role == role)
            if (len(matches) != 1 or type(maximum_bytes) is not int
                    or matches[0].descriptor.size_bytes > maximum_bytes):
                raise ValueError
            entry = matches[0]
            read_request = DockerArtifactReadRequestV1(
                labels, inventory.content_digest, role, entry.relative_path,
                maximum_bytes, entry.descriptor.size_bytes, entry.descriptor.sha256,
                entry.file_identity_digest, request.source_revision,
                inventory.profile_digest, inventory.prepared_plan_digest,
                inventory.artifact_contract_digest, inventory.artifact_root_ref,
            )
        except Exception:
            raise DockerProviderError(DockerDiagnosticCodeV1.BOUNDS_EXCEEDED) from None

        def checked_stream():
            digest = hashlib.sha256()
            total = 0; sequence = 0; eof_seen = False
            try:
                stream = self._reader.iter_artifact_events(read_request)
                for event in stream:
                    if eof_seen:
                        raise ValueError
                    if type(event) is DockerArtifactChunkV1:
                        event = _rebuilt(event, DockerArtifactChunkV1)
                        if (event.stream_digest, event.sequence, event.offset) != (
                                read_request.digest, sequence, total):
                            raise ValueError
                        chunk = event.data
                        total += len(chunk); sequence += 1
                        if total > maximum_bytes or total > entry.descriptor.size_bytes:
                            raise ValueError
                        digest.update(chunk)
                        yield chunk
                        continue
                    if type(event) is not DockerArtifactEOFV1:
                        raise ValueError
                    event = _rebuilt(event, DockerArtifactEOFV1)
                    eof_seen = True
                    if (self._authority.authenticate_eof(event) is not True
                            or (event.stream_digest, event.next_sequence, event.total_bytes,
                                event.sha256, event.file_identity_digest) != (
                                    read_request.digest, sequence, total,
                                    entry.descriptor.sha256, entry.file_identity_digest)):
                        raise ValueError
                if (not eof_seen or total != entry.descriptor.size_bytes
                        or digest.hexdigest() != entry.descriptor.sha256):
                    raise ValueError
            except DockerProviderError:
                raise
            except Exception:
                raise DockerProviderError(DockerDiagnosticCodeV1.MALFORMED_EVIDENCE) from None

        return checked_stream()


class DockerProviderReaderFactoryV1:
    def __init__(self, profile, reader):
        self._profile, self._reader = profile, reader

    def create(self, request):
        try:
            p = self._profile
            expected = (p.provider, p.descriptor.descriptor_digest, p.profile_digest,
                        p.scope.account_ref, p.scope.namespace_ref)
            actual = (request.provider, request.provider_descriptor_digest,
                      request.profile_digest, request.account_ref, request.namespace_ref)
            if type(request) is not ProviderReaderFactoryRequestV1 or actual != expected:
                raise ValueError
            return ResolvedProviderReaderV1(request.request_digest, *actual, self._reader)
        except Exception:
            raise DockerProviderError(DockerDiagnosticCodeV1.BINDING_MISMATCH) from None


__all__: list[str] = []
