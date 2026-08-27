from types import SimpleNamespace
import hashlib

import pytest

from synaptic_tuner.api.v1.results import VerifiedArtifact
from synaptic_tuner.api.v1.runs_facade import RunLogEntry, RunLogLevel
from tuner.execution.coordinator_v1.model import (
    AuthenticatedProviderLogPageV1, AuthenticatedProviderRunObservationV1,
    ProviderLogQueryV1,
)
from tuner.execution.foundation_v2.references import ScopedProviderRunRefV1
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest
from tuner.execution.providers.docker_provider_v1.model import (
    AuthenticatedDockerArtifactInventoryV1, AuthenticatedDockerLogPageV1,
    DockerArtifactChunkV1, DockerArtifactEOFV1, DockerArtifactEntryV1,
    DockerArtifactInventoryV1, DockerCommandBindingV1, DockerEffectIdentityV1,
    DockerLogPageV1, DockerLogReadRequestV1, DockerLookupDispositionV1, DockerLookupResultV1,
    DockerLogTerminalPhaseV1, DockerProviderError, DockerRunPhaseV1,
    PreparedDockerPlanV1, labels_for,
)
from tuner.execution.providers.docker_provider_v1.reader import DockerProviderRunReaderV1
from tests.execution.providers.docker_provider_v1.conftest import D


class ReadPort:
    def __init__(self, labels, content):
        self.labels, self.content = labels, content
        self.descriptor = VerifiedArtifact("result", hashlib.sha256(content).hexdigest(), len(content))
        self.trace = []
        self.chunk_sequence_delta = 0
        self.omit_eof = False
        self.extra_after_eof = False
        self.phase = DockerRunPhaseV1.RUNNING
        self.log_generation_delta = 0
        self.inventory_generation_delta = 0
        self.log_transform = None
        self.inventory_transform = None
        self.eof_transform = None
        self.log_terminal_phase = None
        self.log_stream_digest_override = None
        self.log_entries = (
            RunLogEntry(1, "2026-08-27T12:00:00Z", RunLogLevel.INFO, "queued", "q", 1),
            RunLogEntry(2, "2026-08-27T12:00:01Z", RunLogLevel.INFO, "running", "r", 1),
        )

    def lookup(self, request):
        self.trace.append("lookup")
        return DockerLookupResultV1(
            DockerLookupDispositionV1.FOUND, self.labels,
            "container-1", self.phase,
        )

    def entries(self):
        return self.log_entries

    def logs(self, request):
        self.trace.append("logs")
        self.last_log_request = request
        all_entries = self.entries()
        available = tuple(value for value in all_entries
                          if request.after_sequence is None or value.sequence > request.after_sequence)
        entries = available[:request.limit]
        truncated = len(entries) < len(available)
        generation = request.generation + self.log_generation_delta
        cursor_floor = 0 if request.after_sequence is None else request.after_sequence
        high_watermark = all_entries[-1].sequence if all_entries else 0
        stream_digest = domain_digest(
            "synaptic-docker-log-stream/v1",
            canonical_bytes({"labels_digest": self.labels.digest}),
        )
        if self.log_stream_digest_override is not None:
            stream_digest = self.log_stream_digest_override
        content = DockerLogPageV1(
            request.digest, request.labels.digest, stream_digest, request.query_digest,
            generation, request.after_sequence,
            request.limit, request.maximum_bytes, request.maximum_entry_bytes,
            entries, (entries[0].sequence if entries else None),
            (entries[-1].sequence if entries else None), not truncated, truncated,
            (entries[-1].sequence if truncated else None), high_watermark,
            self.log_terminal_phase,
            (generation if self.log_terminal_phase is not None else None), D[7],
        )
        if self.log_transform is not None:
            content = self.log_transform(content)
        self.last_log_page = content
        return AuthenticatedDockerLogPageV1(content, "docker-test", "key-v1", D[8])

    def artifact_inventory(self, request):
        self.trace.append("inventory")
        content = DockerArtifactInventoryV1(
            request.labels, request.digest,
            request.generation + self.inventory_generation_delta,
            request.profile_digest, request.prepared_plan_digest,
            request.artifact_contract_digest, request.artifact_root_ref,
            (DockerArtifactEntryV1(self.descriptor, "result.json", D[6]),), D[9],
        )
        if self.inventory_transform is not None:
            content = self.inventory_transform(content)
        return AuthenticatedDockerArtifactInventoryV1(content, "docker-test", "key-v1", D[10])

    def iter_artifact_events(self, request):
        self.trace.append("events")
        yield DockerArtifactChunkV1(request.digest, self.chunk_sequence_delta, 0, self.content)
        if not self.omit_eof:
            eof = DockerArtifactEOFV1(
                request.digest, 1, len(self.content), self.descriptor.sha256,
                D[6], D[11], "docker-test", "key-v1", D[12],
            )
            if self.eof_transform is not None:
                eof = self.eof_transform(eof)
            yield eof
        if self.extra_after_eof:
            yield DockerArtifactChunkV1(request.digest, 1, len(self.content), b"x")


class BypassReader(DockerProviderRunReaderV1):
    def __init__(self, binding, ref, labels, read_port, authority):
        self._bound = (binding, ref, labels)
        self._reader = read_port
        self._authority = authority
        self._observed_at = "2026-08-27T12:00:00Z"
        self._log_state_lock = __import__("threading").Lock()
        self._log_states = {}
        self._log_page_commitments = {}
    def _validate(self, request, purpose):
        return self._bound


class Authority:
    def __init__(self):
        self.log_valid = True
        self.inventory_valid = True
        self.eof_valid = True
    def observation(self, content):
        return AuthenticatedProviderRunObservationV1(content, "docker-evidence", "key-v1", "a" * 64)
    def log_page(self, content):
        return AuthenticatedProviderLogPageV1(content, "docker-evidence", "key-v1", "b" * 64)
    def authenticate_log_page(self, value): return self.log_valid and type(value) is AuthenticatedDockerLogPageV1
    def authenticate_inventory(self, value): return self.inventory_valid and type(value) is AuthenticatedDockerArtifactInventoryV1
    def authenticate_eof(self, value): return self.eof_valid and type(value) is DockerArtifactEOFV1
    def authenticate(self, value):
        return type(value) in {AuthenticatedProviderRunObservationV1, AuthenticatedProviderLogPageV1}


def stack(profile, plan, run, content=b"fixture-result"):
    prepared = PreparedDockerPlanV1(
        profile, run.project_ref, run.run_id, plan.plan_fingerprint,
        plan.basis.source_digest, D[14],
    )
    binding = DockerCommandBindingV1(DockerEffectIdentityV1(
        D[13], "submit-effect", "submit", prepared
    ), b"{}")
    labels = labels_for(binding.identity)
    ref = ScopedProviderRunRefV1("docker", profile.provider.profile_ref, "account", "namespace", "container-1")
    port = ReadPort(labels, content)
    reader = BypassReader(binding, ref, labels, port, Authority())
    request = SimpleNamespace(
        run=run, request_digest=D[0], source_workflow_record_digest=D[1],
        source_revision=1, provider_run=SimpleNamespace(binding_digest=D[2]),
    )
    return reader, port, request


def forged(value, **changes):
    for name, replacement in changes.items():
        object.__setattr__(value, name, replacement)
    return value


def test_artifact_manifest_and_stream_accept_only_authenticated_exact_eof(profile, plan, run):
    reader, port, request = stack(profile, plan, run)
    manifest = reader.artifacts(request)
    assert b"".join(reader.iter_artifact_bytes(request, manifest, "result", maximum_bytes=1024)) == b"fixture-result"
    assert port.trace == ["inventory", "inventory", "events"]


@pytest.mark.parametrize("mode", ("truncated", "extra", "tampered", "reordered"))
def test_artifact_stream_closes_every_hostile_event_case(profile, plan, run, mode):
    reader, port, request = stack(profile, plan, run)
    manifest = reader.artifacts(request)
    if mode == "truncated": port.omit_eof = True
    elif mode == "extra": port.extra_after_eof = True
    elif mode == "tampered": port.content = b"tampered-data"
    else: port.chunk_sequence_delta = 1
    with pytest.raises(DockerProviderError) as caught:
        list(reader.iter_artifact_bytes(request, manifest, "result", maximum_bytes=1024))
    assert str(caught.value) == "docker_malformed_evidence"


@pytest.mark.parametrize("mode", (
    "labels", "request", "generation", "profile", "plan", "contract", "root",
    "role",
))
def test_artifact_inventory_rejects_cross_binding_substitution(profile, plan, run, mode):
    reader, port, request = stack(profile, plan, run)
    def transform(inventory):
        if mode == "labels":
            other = DockerEffectIdentityV1(D[4], "other-submit", "submit", reader._bound[0].plan)
            return forged(inventory, labels=labels_for(other))
        if mode == "request": return forged(inventory, request_digest=D[4])
        if mode == "generation": return forged(inventory, generation=2)
        if mode == "profile": return forged(inventory, profile_digest=D[4])
        if mode == "plan": return forged(inventory, prepared_plan_digest=D[4])
        if mode == "contract": return forged(inventory, artifact_contract_digest=D[4])
        if mode == "root": return forged(inventory, artifact_root_ref="other-root")
        entry = inventory.entries[0]
        descriptor = VerifiedArtifact("other", entry.descriptor.sha256, entry.descriptor.size_bytes)
        replacement = DockerArtifactEntryV1(descriptor, entry.relative_path, entry.file_identity_digest)
        return forged(inventory, entries=(replacement,))
    port.inventory_transform = transform
    with pytest.raises(DockerProviderError) as caught:
        reader.artifacts(request)
    assert str(caught.value) == "docker_malformed_evidence"


@pytest.mark.parametrize("mode", ("stream_digest", "file_identity", "size", "sha"))
def test_artifact_eof_rejects_cross_request_and_file_substitution(profile, plan, run, mode):
    reader, port, request = stack(profile, plan, run)
    manifest = reader.artifacts(request)
    def transform(eof):
        if mode == "stream_digest": return forged(eof, stream_digest=D[4])
        if mode == "file_identity": return forged(eof, file_identity_digest=D[4])
        if mode == "size": return forged(eof, total_bytes=eof.total_bytes + 1)
        return forged(eof, sha256=D[4])
    port.eof_transform = transform
    with pytest.raises(DockerProviderError) as caught:
        list(reader.iter_artifact_bytes(request, manifest, "result", maximum_bytes=1024))
    assert str(caught.value) == "docker_malformed_evidence"


def test_reader_rejects_before_external_io_when_request_is_unauthenticated(profile):
    class Deny:
        def authenticate(self, request): return False
    class Explode:
        def __getattr__(self, name): raise AssertionError("external reader invoked")
    reader = DockerProviderRunReaderV1(
        profile, Explode(), Explode(), Deny(), Explode(), Explode(),
        observed_at="2026-08-27T12:00:00Z",
    )
    with pytest.raises(DockerProviderError) as caught:
        reader.observe(object())
    assert str(caught.value) == "docker_authentication_failed"


def test_observe_returns_closed_authenticated_normalized_evidence(profile, plan, run):
    reader, port, request = stack(profile, plan, run)
    result = reader.observe(request)
    assert type(result) is AuthenticatedProviderRunObservationV1
    assert result.content.phase.value == "running"
    assert port.trace == ["lookup"]


def test_logs_are_authenticated_cursor_generation_bound_and_strictly_bounded(profile, plan, run):
    reader, port, request = stack(profile, plan, run)
    query = ProviderLogQueryV1(1, 1, 4096)
    result = reader.logs(request, query)
    assert type(result) is AuthenticatedProviderLogPageV1
    assert tuple(entry.sequence for entry in result.content.entries) == (2,)
    assert result.content.total_bytes == 1
    assert port.trace == ["logs"]


@pytest.mark.parametrize("mode", (
    "query", "generation", "limit", "maximum_bytes", "entry_bound",
    "gap", "overlap", "replayed_cursor", "empty_incomplete", "empty_truncated",
))
def test_logs_reject_every_hostile_cursor_and_page_shape(profile, plan, run, mode):
    reader, port, request = stack(profile, plan, run)
    def transform(page):
        if mode == "query": return forged(page, query_digest=D[4])
        if mode == "generation": return forged(page, generation=2)
        if mode == "limit": return forged(page, requested_limit=2)
        if mode == "maximum_bytes": return forged(page, requested_maximum_bytes=8192)
        if mode == "entry_bound": return forged(page, maximum_entry_bytes=2048)
        if mode in {"gap", "overlap"}:
            sequence = 3 if mode == "gap" else 1
            entry = RunLogEntry(sequence, "2026-08-27T12:00:02Z", RunLogLevel.INFO, "hostile", "x", 1)
            return forged(page, entries=(entry,), first_sequence=sequence, last_sequence=sequence)
        if mode == "replayed_cursor": return forged(page, after_sequence=0)
        if mode == "empty_incomplete":
            return forged(page, entries=(), first_sequence=None, last_sequence=None, complete=False)
        return forged(
            page, entries=(), first_sequence=None, last_sequence=None,
            complete=False, truncated=True, next_sequence=None,
        )
    port.log_transform = transform
    with pytest.raises(DockerProviderError) as caught:
        reader.logs(request, ProviderLogQueryV1(1, 1, 4096))
    assert str(caught.value) == "docker_malformed_evidence"


def test_reader_and_lazy_iterator_close_transport_exception_text(profile, plan, run):
    reader, port, request = stack(profile, plan, run)
    manifest = reader.artifacts(request)
    secret = "raw-secret-sentinel"
    def exploding(read_request):
        yield DockerArtifactChunkV1(read_request.digest, 0, 0, port.content[:1])
        raise RuntimeError(secret)
    port.iter_artifact_events = exploding
    stream = reader.iter_artifact_bytes(request, manifest, "result", maximum_bytes=1024)
    assert next(stream) == port.content[:1]
    with pytest.raises(DockerProviderError) as caught:
        next(stream)
    assert str(caught.value) == "docker_malformed_evidence"
    assert secret not in str(caught.value)


@pytest.mark.parametrize("mode", (
    "hidden_high_watermark", "terminal_without_generation",
    "false_complete", "truncated_terminal_generation",
))
def test_log_high_watermark_and_terminal_generation_prevent_hidden_truncation(profile, plan, run, mode):
    reader, port, request = stack(profile, plan, run)
    query = ProviderLogQueryV1(None, 1, 4096)
    def transform(page):
        if mode == "hidden_high_watermark":
            return forged(
                page, complete=True, truncated=False, next_sequence=None,
                high_watermark_sequence=page.last_sequence + 1,
                terminal_generation=page.generation,
            )
        if mode == "terminal_without_generation":
            return forged(
                page, complete=True, truncated=False, next_sequence=None,
                high_watermark_sequence=page.last_sequence,
                terminal_phase=DockerLogTerminalPhaseV1.SUCCEEDED,
                terminal_generation=None,
            )
        if mode == "false_complete":
            return forged(
                page, complete=True, truncated=False, next_sequence=None,
                terminal_generation=page.generation,
            )
        return forged(page, terminal_generation=page.generation)
    port.log_transform = transform
    with pytest.raises(DockerProviderError) as caught:
        reader.logs(request, query)
    assert str(caught.value) == "docker_malformed_evidence"


def test_nonterminal_log_stream_can_grow_without_cursor_relative_watermark(profile, plan, run):
    reader, port, request = stack(profile, plan, run)
    first = reader.logs(request, ProviderLogQueryV1(None, 2, 4096))
    assert tuple(value.sequence for value in first.content.entries) == (1, 2)
    port.log_entries += (
        RunLogEntry(3, "2026-08-27T12:00:02Z", RunLogLevel.INFO, "running", "g", 1),
    )
    second = reader.logs(request, ProviderLogQueryV1(2, 2, 4096))
    assert tuple(value.sequence for value in second.content.entries) == (3,)


def test_log_cursor_beyond_atomic_stream_snapshot_is_rejected(profile, plan, run):
    reader, _, request = stack(profile, plan, run)
    with pytest.raises(DockerProviderError) as caught:
        reader.logs(request, ProviderLogQueryV1(3, 2, 4096))
    assert str(caught.value) == "docker_malformed_evidence"


def test_same_stream_generation_high_watermark_cannot_regress(profile, plan, run):
    reader, port, request = stack(profile, plan, run)
    reader.logs(request, ProviderLogQueryV1(None, 2, 4096))
    port.log_entries = port.log_entries[:1]
    with pytest.raises(DockerProviderError) as caught:
        reader.logs(request, ProviderLogQueryV1(None, 2, 4096))
    assert str(caught.value) == "docker_malformed_evidence"


@pytest.mark.parametrize("terminal", (False, True))
def test_log_high_watermark_cannot_regress_across_generations(profile, plan, run, terminal):
    reader, port, request = stack(profile, plan, run)
    if terminal:
        port.log_terminal_phase = DockerLogTerminalPhaseV1.SUCCEEDED
    reader.logs(request, ProviderLogQueryV1(None, 2, 4096))
    request.source_revision = 2
    port.log_entries = port.log_entries[:1]
    with pytest.raises(DockerProviderError) as caught:
        reader.logs(request, ProviderLogQueryV1(None, 2, 4096))
    assert str(caught.value) == "docker_malformed_evidence"
    assert port.last_log_request.generation == port.last_log_page.generation == 2


@pytest.mark.parametrize("growth", ("equal", "larger"))
def test_nonterminal_cross_generation_equal_or_larger_watermark_is_accepted(profile, plan, run, growth):
    reader, port, request = stack(profile, plan, run)
    if growth == "larger":
        port.log_entries = port.log_entries[:1]
    first = reader.logs(request, ProviderLogQueryV1(None, 2, 4096))
    request.source_revision = 2
    if growth == "larger":
        port.log_entries += (
            RunLogEntry(
                2, "2026-08-27T12:00:01Z", RunLogLevel.INFO,
                "running", "r", 1,
            ),
        )
    second = reader.logs(request, ProviderLogQueryV1(None, 2, 4096))
    assert tuple(value.sequence for value in first.content.entries) == (
        (1,) if growth == "larger" else (1, 2)
    )
    assert tuple(value.sequence for value in second.content.entries) == (1, 2)


def test_terminal_stream_remains_immutable_across_correctly_bound_generation(profile, plan, run):
    reader, port, request = stack(profile, plan, run)
    port.log_terminal_phase = DockerLogTerminalPhaseV1.SUCCEEDED
    reader.logs(request, ProviderLogQueryV1(None, 2, 4096))
    request.source_revision = 2
    with pytest.raises(DockerProviderError) as caught:
        reader.logs(request, ProviderLogQueryV1(None, 2, 4096))
    assert str(caught.value) == "docker_malformed_evidence"


def test_same_stream_generation_query_rejects_changed_sequence_content(profile, plan, run):
    reader, port, request = stack(profile, plan, run)
    query = ProviderLogQueryV1(None, 2, 4096)
    reader.logs(request, query)
    original = port.log_entries[0]
    port.log_entries = (
        RunLogEntry(
            original.sequence, original.timestamp, original.level,
            "forged", "x", 1,
        ),
        port.log_entries[1],
    )
    with pytest.raises(DockerProviderError) as caught:
        reader.logs(request, query)
    assert str(caught.value) == "docker_malformed_evidence"


def test_log_page_commitment_cache_overflow_fails_closed(profile, plan, run):
    reader, _, request = stack(profile, plan, run)
    reader._log_page_commitments = {
        (D[4], generation, D[5]): D[6] for generation in range(1, 1025)
    }
    states_before = dict(reader._log_states)
    commitments_before = dict(reader._log_page_commitments)
    with pytest.raises(DockerProviderError) as caught:
        reader.logs(request, ProviderLogQueryV1(None, 2, 4096))
    assert str(caught.value) == "docker_malformed_evidence"
    assert reader._log_states == states_before
    assert reader._log_page_commitments == commitments_before


def test_terminal_log_snapshot_is_stable_across_multiple_pages(profile, plan, run):
    reader, port, request = stack(profile, plan, run)
    port.log_terminal_phase = DockerLogTerminalPhaseV1.SUCCEEDED
    first = reader.logs(request, ProviderLogQueryV1(None, 1, 4096))
    second = reader.logs(request, ProviderLogQueryV1(1, 1, 4096))
    assert first.content.truncated is True
    assert second.content.truncated is False
    port.log_terminal_phase = DockerLogTerminalPhaseV1.FAILED
    with pytest.raises(DockerProviderError):
        reader.logs(request, ProviderLogQueryV1(2, 1, 4096))


def test_log_stream_digest_substitution_is_rejected(profile, plan, run):
    reader, port, request = stack(profile, plan, run)
    port.log_stream_digest_override = D[4]
    with pytest.raises(DockerProviderError) as caught:
        reader.logs(request, ProviderLogQueryV1(None, 2, 4096))
    assert str(caught.value) == "docker_malformed_evidence"


@pytest.mark.parametrize("boundary", (
    "logs_call", "logs_auth", "inventory_call", "inventory_auth", "eof_auth",
))
def test_reader_closes_every_injected_port_and_authenticator_exception(profile, plan, run, boundary):
    reader, port, request = stack(profile, plan, run)
    secret = "raw-secret-sentinel"
    def explode(*args, **kwargs):
        raise RuntimeError(secret)
    if boundary == "logs_call":
        port.logs = explode
        call = lambda: reader.logs(request, ProviderLogQueryV1(None, 2, 4096))
    elif boundary == "logs_auth":
        reader._authority.authenticate_log_page = explode
        call = lambda: reader.logs(request, ProviderLogQueryV1(None, 2, 4096))
    elif boundary == "inventory_call":
        port.artifact_inventory = explode
        call = lambda: reader.artifacts(request)
    elif boundary == "inventory_auth":
        reader._authority.authenticate_inventory = explode
        call = lambda: reader.artifacts(request)
    else:
        manifest = reader.artifacts(request)
        reader._authority.authenticate_eof = explode
        call = lambda: list(reader.iter_artifact_bytes(request, manifest, "result", maximum_bytes=1024))
    with pytest.raises(DockerProviderError) as caught:
        call()
    assert str(caught.value) == "docker_malformed_evidence"
    assert secret not in str(caught.value)


@pytest.mark.parametrize("failure", ("log_generation", "log_signer", "inventory_generation", "inventory_signer", "eof_signer"))
def test_authenticated_read_evidence_fails_closed_on_binding_or_signer_change(profile, plan, run, failure):
    reader, port, request = stack(profile, plan, run)
    if failure == "log_generation":
        port.log_generation_delta = 1
        call = lambda: reader.logs(request, ProviderLogQueryV1(None, 2, 4096))
    elif failure == "log_signer":
        reader._authority.log_valid = False
        call = lambda: reader.logs(request, ProviderLogQueryV1(None, 2, 4096))
    elif failure == "inventory_generation":
        port.inventory_generation_delta = 1
        call = lambda: reader.artifacts(request)
    elif failure == "inventory_signer":
        reader._authority.inventory_valid = False
        call = lambda: reader.artifacts(request)
    else:
        manifest = reader.artifacts(request)
        reader._authority.eof_valid = False
        call = lambda: list(reader.iter_artifact_bytes(request, manifest, "result", maximum_bytes=1024))
    with pytest.raises(DockerProviderError) as caught:
        call()
    assert str(caught.value) == "docker_malformed_evidence"
