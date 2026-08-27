"""Focused conformance tests for the provider-neutral publication kernel."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
from threading import Barrier, Lock, Thread
from threading import Event

import pytest

from synaptic_tuner.api.v1.artifacts_facade import (
    ArtifactsOperations,
    PublicationRef,
    PublicationRequest,
    PublicationState,
)
from synaptic_tuner.api.v1.results import TrainingRunRef, VerifiedArtifact
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest
from tuner.execution.coordinator_v1.publication import (
    AuthenticatedDestinationV1,
    AuthenticatedDestinationInventoryV1,
    AuthenticatedLookupV1,
    AuthenticatedPublicationReceiptV1,
    AuthenticatedPublicationTombstoneV1,
    AuthenticatedVerifiedSourceV1,
    DestinationArtifactV1,
    DestinationInventoryV1,
    LookupOutcomeV1,
    LookupRecoveryPermitV1,
    PublicationCodeV1,
    PublicationCommandV1,
    PublicationErrorV1,
    PublicationEventKindV1,
    PublicationOperationsV1,
    PublicationPhaseV1,
    PublicationRecordV1,
    StrongInMemoryPublicationStoreV1,
    TransferOwnershipV1,
)


RUN = TrainingRunRef("run-1", "project-1")
KEY = "host-key"
ZERO = "0" * 64


class Authority:
    @staticmethod
    def sign(purpose, payload, key_ref=KEY):
        return domain_digest(
            "test-evidence-tag/v1",
            canonical_bytes({
                "purpose": purpose, "key_ref": key_ref,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
            }),
        )

    def verify(self, purpose, payload, tag, key_ref):
        return tag == self.sign(purpose, payload, key_ref)


AUTH = Authority()


def _artifact(role="adapter", data=b"adapter"):
    return VerifiedArtifact(role, hashlib.sha256(data).hexdigest(), len(data)), data


def _destination(ref="opaque/local-like", display="Local target", config=b"local",
                 maximum_artifact_bytes=1_048_576,
                 maximum_total_bytes=8_388_608):
    value = AuthenticatedDestinationV1(
        "synaptic-publication-destination/v1", ref, display,
        hashlib.sha256(config).hexdigest(), hashlib.sha256(b"create-only").hexdigest(),
        maximum_artifact_bytes, maximum_total_bytes, "host-authority", KEY, ZERO,
    )
    return replace(value, tag=AUTH.sign("publication-destination/v1", value.payload))


def _source(artifacts, run=RUN):
    value = AuthenticatedVerifiedSourceV1(
        "synaptic-publication-verified-source/v1", run,
        tuple(item for item, _ in artifacts), hashlib.sha256(b"verified").hexdigest(),
        "host-authority", KEY, ZERO,
    )
    return replace(value, tag=AUTH.sign("publication-verified-source/v1", value.payload))


class Stream:
    def __init__(self, run, artifact, data, *, chunks=None, fail_after=False):
        self.run, self.artifact = run, artifact
        self.maximum_bytes = max(1, artifact.size_bytes)
        self._data, self._chunks, self._fail_after = data, chunks, fail_after

    def iter_bytes(self):
        chunks = self._chunks if self._chunks is not None else (() if not self._data else (self._data,))
        yield from chunks
        if self._fail_after:
            raise RuntimeError("secret provider traceback")


class Sources:
    def __init__(self, artifacts, *, descriptor=None, stream_factory=None):
        self.artifacts = artifacts
        self.descriptor = descriptor or _source(artifacts)
        self.stream_factory = stream_factory
        self.opens = 0

    def describe(self, run):
        return self.descriptor

    def open(self, request):
        self.opens += 1
        artifact, data = next(item for item in self.artifacts if item[0].role == request.role)
        if self.stream_factory:
            return self.stream_factory(request.run, artifact, data)
        return Stream(request.run, artifact, data)


class Sink:
    def __init__(self, owner, key):
        self.owner, self.key, self.data, self.aborted = owner, key, bytearray(), False

    def write(self, chunk):
        self.data.extend(chunk)

    def finish(self):
        self.owner.data[self.key] = bytes(self.data)
        return "spool/" + self.key.replace(":", "/")

    def abort(self):
        self.aborted = True


class Spool:
    def __init__(self):
        self.data = {}

    def open(self, publication_id, role, maximum_bytes):
        return Sink(self, publication_id + ":" + role)


def _receipt(command, ownership, inventory):
    inventory_evidence = AuthenticatedDestinationInventoryV1(
        inventory, command.publication_id, command.command_digest,
        command.mutation_id, ownership.ownership_id,
        "2026-08-27T12:00:01Z", command.destination_authority_ref,
        command.destination_key_ref, ZERO,
    )
    inventory_evidence = replace(
        inventory_evidence,
        tag=AUTH.sign("publication-destination-inventory/v1",
                      inventory_evidence.payload),
    )
    value = AuthenticatedPublicationReceiptV1(
        "synaptic-publication-receipt/v1", command.publication_id,
        command.command_digest, command.run, command.source_identity_digest,
        command.destination_ref, command.destination_identity_digest,
        command.mutation_id, ownership.claim_digest, ownership.ownership_id,
        inventory_evidence, "2026-08-27T12:00:01Z",
        command.destination_authority_ref, command.destination_key_ref, ZERO,
    )
    return replace(value, tag=AUTH.sign("publication-receipt/v1", value.payload))


def _tombstone(command, permit, mutation_registry_digest):
    value = AuthenticatedPublicationTombstoneV1(
        "synaptic-publication-tombstone/v1", command.publication_id,
        command.mutation_id, command.command_digest, permit.claim_digest,
        command.destination_ref, command.destination_identity_digest,
        command.destination_configuration_digest, command.destination_policy_digest,
        command.destination_authority_ref, command.destination_key_ref,
        permit.fenced_ownership_id, permit.permit_id, mutation_registry_digest,
        "2026-08-27T12:00:02Z", hashlib.sha256(b"absent").hexdigest(),
        command.destination_authority_ref, command.destination_key_ref, ZERO,
    )
    return replace(value, tag=AUTH.sign("publication-tombstone/v1", value.payload))


def _lookup(outcome, command, permit, receipt=None, *, absent=False, bind=True):
    registry = hashlib.sha256(b"mutation-registry").hexdigest()
    tombstone = _tombstone(command, permit, registry) if absent else None
    value = AuthenticatedLookupV1(
        "synaptic-publication-lookup/v1", outcome,
        command.publication_id if bind else "f" * 64,
        command.command_digest, command.destination_identity_digest,
        command.mutation_id, permit.fenced_ownership_id, permit.permit_id,
        registry, "2026-08-27T12:00:02Z", tombstone, receipt,
        command.destination_authority_ref, command.destination_key_ref, ZERO,
    )
    return replace(value, tag=AUTH.sign("publication-lookup/v1", value.payload))


class Adapter:
    """One fake implementation; the opaque descriptor is only configuration data."""

    def __init__(self, spool, *, mode="success", barrier=None):
        self.spool, self.mode, self.barrier = spool, mode, barrier
        self.publish_calls = 0
        self.lookup_calls = 0
        self.read_calls = 0
        self.receipt = None
        self.command = None
        self.ownership = None
        self.published = {}
        self.entered = None
        self.release = None
        self._lock = Lock()

    def publish_once(self, command, source, ownership):
        with self._lock:
            self.publish_calls += 1
        self.command, self.ownership = command, ownership
        if self.entered is not None:
            self.entered.set()
            self.release.wait(timeout=5)
        if self.barrier:
            self.barrier.wait(timeout=5)
        for spooled in source.artifacts:
            key = command.publication_id + ":" + spooled.artifact.role
            self.published[spooled.artifact.role] = self.spool.data[key]
        inventory = DestinationInventoryV1(tuple(
            DestinationArtifactV1(item.role, "bundle/" + item.role,
                                  item.sha256, item.size_bytes)
            for item in command.source_inventory
        ))
        self.receipt = _receipt(command, ownership, inventory)
        if self.mode == "raise_after_effect":
            raise RuntimeError("secret provider response")
        if self.mode == "bad_receipt":
            return replace(self.receipt, command_digest="f" * 64)
        if self.mode == "bad_inventory":
            items = self.receipt.inventory.inventory.artifacts
            changed_inventory = replace(
                self.receipt.inventory,
                inventory=DestinationInventoryV1((
                    replace(items[0], size_bytes=items[0].size_bytes + 1),
                ) + items[1:]),
                tag=ZERO,
            )
            changed_inventory = replace(
                changed_inventory,
                tag=AUTH.sign("publication-destination-inventory/v1",
                              changed_inventory.payload),
            )
            changed_receipt = replace(
                self.receipt, inventory=changed_inventory, tag=ZERO,
            )
            return replace(
                changed_receipt,
                tag=AUTH.sign("publication-receipt/v1", changed_receipt.payload),
            )
        return self.receipt

    def lookup(self, command, permit):
        self.lookup_calls += 1
        if self.mode == "lookup_raise":
            raise RuntimeError("secret lookup traceback")
        if self.mode == "absent":
            return _lookup(LookupOutcomeV1.DEFINITELY_ABSENT, command, permit,
                           absent=True)
        if self.mode == "conflict":
            return _lookup(LookupOutcomeV1.CONFLICT, command, permit)
        if self.mode == "indeterminate" or self.receipt is None:
            return _lookup(LookupOutcomeV1.INDETERMINATE, command, permit)
        return _lookup(LookupOutcomeV1.FOUND, command, permit, self.receipt)

    def iter_bytes(self, command, artifact, maximum_bytes):
        self.read_calls += 1
        data = self.published[artifact.role]
        if self.mode == "tamper":
            data = b"x" + data[1:]
        if self.mode == "truncate":
            data = data[:-1]
        if self.mode == "trailing":
            data += b"x"
        if data:
            yield data


class Registry:
    def __init__(self, entries):
        self.entries = dict(entries)
        self.resolve_calls = 0
        self.list_calls = 0

    def resolve(self, destination_ref):
        self.resolve_calls += 1
        return self.entries[destination_ref]

    def list(self, limit):
        self.list_calls += 1
        values = tuple(item[0] for _, item in sorted(self.entries.items()))
        return values[:limit], len(values) <= 100


class Clock:
    def __init__(self):
        self.value = 0

    def __call__(self):
        self.value += 1
        return f"2026-08-27T12:00:{self.value:02d}Z"


def _stack(ref="opaque/local-like", *, mode="success", artifacts=None,
           stream_factory=None, store=None, barrier=None):
    artifacts = artifacts or [_artifact()]
    spool = Spool()
    adapter = Adapter(spool, mode=mode, barrier=barrier)
    descriptor = _destination(ref, "Target " + ref.split("/")[-1], ref.encode())
    registry = Registry({ref: (descriptor, adapter)})
    sources = Sources(artifacts, stream_factory=stream_factory)
    store = store or StrongInMemoryPublicationStoreV1()
    service = PublicationOperationsV1(
        store=store, destinations=registry, sources=sources, spool=spool,
        authority=AUTH, clock=Clock(),
    )
    return service, store, registry, sources, spool, adapter


def test_command_is_canonical_and_every_binding_changes_identity() -> None:
    artifact, _ = _artifact()
    destination = _destination()
    source = _source([(artifact, b"adapter")])
    command = PublicationCommandV1.build(
        run=RUN, source_identity_digest=source.source_identity_digest,
        source_inventory=source.artifacts, destination_ref=destination.destination_ref,
        destination_identity_digest=destination.identity_digest,
        destination_configuration_digest=destination.configuration_digest,
        destination_policy_digest=destination.policy_digest,
        maximum_artifact_bytes=destination.maximum_artifact_bytes,
        maximum_total_bytes=destination.maximum_total_bytes,
        destination_authority_ref=destination.authority_ref,
        destination_key_ref=destination.key_ref,
    )
    assert command == PublicationCommandV1.build(
        run=RUN, source_identity_digest=source.source_identity_digest,
        source_inventory=source.artifacts, destination_ref=destination.destination_ref,
        destination_identity_digest=destination.identity_digest,
        destination_configuration_digest=destination.configuration_digest,
        destination_policy_digest=destination.policy_digest,
        maximum_artifact_bytes=destination.maximum_artifact_bytes,
        maximum_total_bytes=destination.maximum_total_bytes,
        destination_authority_ref=destination.authority_ref,
        destination_key_ref=destination.key_ref,
    )
    changes = (
        dict(run=TrainingRunRef("run-2", "project-1")),
        dict(source_identity_digest="1" * 64),
        dict(source_inventory=(VerifiedArtifact("adapter", "2" * 64, 7),)),
        dict(destination_ref="opaque/other"),
        dict(destination_identity_digest="3" * 64),
    )
    base = dict(run=RUN, source_identity_digest=source.source_identity_digest,
                source_inventory=source.artifacts,
                destination_ref=destination.destination_ref,
                destination_identity_digest=destination.identity_digest,
                destination_configuration_digest=destination.configuration_digest,
                destination_policy_digest=destination.policy_digest,
                maximum_artifact_bytes=destination.maximum_artifact_bytes,
                maximum_total_bytes=destination.maximum_total_bytes,
                destination_authority_ref=destination.authority_ref,
                destination_key_ref=destination.key_ref)
    assert len({PublicationCommandV1.build(**(base | change)).publication_id
                for change in changes} | {command.publication_id}) == 6
    with pytest.raises(ValueError):
        replace(command, mutation_id="4" * 64)


def test_strong_store_exact_claim_replay_and_descendant_cas() -> None:
    artifact, _ = _artifact()
    source = _source([(artifact, b"adapter")])
    destination = _destination()
    command = PublicationCommandV1.build(
        run=RUN, source_identity_digest=source.source_identity_digest,
        source_inventory=source.artifacts, destination_ref=destination.destination_ref,
        destination_identity_digest=destination.identity_digest,
        destination_configuration_digest=destination.configuration_digest,
        destination_policy_digest=destination.policy_digest,
        maximum_artifact_bytes=destination.maximum_artifact_bytes,
        maximum_total_bytes=destination.maximum_total_bytes,
        destination_authority_ref=destination.authority_ref,
        destination_key_ref=destination.key_ref,
    )
    record = PublicationRecordV1.claim(command, "2026-08-27T12:00:00Z")
    store = StrongInMemoryPublicationStoreV1()
    assert store.claim(record) == (record, True)
    assert store.claim(record) == (record, False)
    admission = store.begin_transfer(
        command.publication_id, record.record_digest, "2026-08-27T12:00:01Z"
    )
    admitted = admission.record
    assert tuple(event.kind for event in admitted.history) == (
        PublicationEventKindV1.CLAIMED,
        PublicationEventKindV1.TRANSFER_ADMITTED,
    )
    with pytest.raises(PublicationErrorV1) as error:
        admitted.transition(PublicationPhaseV1.VERIFIED, "2026-08-27T12:00:02Z")
    assert error.value.code is PublicationCodeV1.STATE_CONFLICT


@pytest.mark.parametrize("ref", ["opaque/local-like", "opaque/hf-repo-like"])
def test_same_provider_neutral_conformance_for_arbitrary_destinations(ref) -> None:
    service, store, registry, sources, spool, adapter = _stack(ref)
    result = service.publish(PublicationRequest(RUN, ref))
    assert result.state is PublicationState.VERIFIED
    assert result.artifacts[0].role == "adapter"
    assert adapter.publish_calls == adapter.read_calls == 1
    assert adapter.lookup_calls == 0
    assert service.publish(PublicationRequest(RUN, ref)) == result
    assert adapter.publish_calls == 1
    assert service.verify(result.publication).verified


def _normalized_lifecycle_trace(ref, scenario):
    initial_mode = "raise_after_effect" if scenario in {
        "found", "absent", "indeterminate", "conflict"
    } else scenario
    if scenario == "success":
        initial_mode = "success"
    service, store, _, _, _, adapter = _stack(ref, mode=initial_mode)
    request = PublicationRequest(RUN, ref)
    first = service.publish(request)
    states = [first.state.value]
    verified = None
    if scenario == "success":
        states.append(service.publish(request).state.value)
        verified = service.verify(first.publication).verified
    elif scenario in {"found", "absent", "indeterminate", "conflict"}:
        adapter.mode = scenario if scenario != "found" else "success"
        states.append(service.publish(request).state.value)
    record = store.get(first.publication.publication_id)
    return {
        "states": tuple(states),
        "phase": record.phase.value,
        "verified": verified,
        "publish_calls": adapter.publish_calls,
        "lookup_calls": adapter.lookup_calls,
        "read_calls": adapter.read_calls,
        "ownership_count": len(record.ownership_history),
        "permit_count": len(record.recovery_permits),
    }


@pytest.mark.parametrize(
    "scenario",
    [
        "success", "found", "absent", "indeterminate", "conflict",
        "bad_receipt", "bad_inventory", "tamper", "truncate", "trailing",
    ],
)
def test_two_opaque_profiles_have_identical_full_lifecycle_traces(scenario) -> None:
    traces = tuple(
        _normalized_lifecycle_trace(ref, scenario)
        for ref in ("opaque/local-like", "opaque/registry-like")
    )
    assert traces[0] == traces[1]


def test_source_requires_authentication_and_complete_eof_before_admission() -> None:
    artifact, data = _artifact()
    bad = replace(_source([(artifact, data)]), tag="f" * 64)
    service, store, _, sources, _, adapter = _stack()
    sources.descriptor = bad
    with pytest.raises(PublicationErrorV1) as error:
        service.publish(PublicationRequest(RUN, "opaque/local-like"))
    assert error.value.code is PublicationCodeV1.SOURCE_UNVERIFIED
    assert adapter.publish_calls == 0
    assert store.list("opaque/local-like", 101)[0] == ()

    service, store, _, _, _, adapter = _stack(
        stream_factory=lambda run, item, raw: Stream(run, item, raw,
                                                     chunks=(raw[:-1],))
    )
    with pytest.raises(PublicationErrorV1) as error:
        service.publish(PublicationRequest(RUN, "opaque/local-like"))
    assert error.value.code is PublicationCodeV1.SOURCE_CONTENT_INVALID
    assert adapter.publish_calls == 0
    records = store.list("opaque/local-like", 101)[0]
    assert len(records) == 1
    assert records[0].phase is PublicationPhaseV1.FAILED_BEFORE_EFFECT


def _forge_verified_source(artifacts, *, tag=ZERO):
    value = object.__new__(AuthenticatedVerifiedSourceV1)
    fields = {
        "schema_version": "synaptic-publication-verified-source/v1",
        "run": RUN,
        "artifacts": tuple(artifacts),
        "verification_digest": hashlib.sha256(b"verified").hexdigest(),
        "authority_ref": "host-authority",
        "key_ref": KEY,
        "tag": tag,
    }
    for name, field_value in fields.items():
        object.__setattr__(value, name, field_value)
    return value


def test_oversized_or_malformed_source_evidence_is_closed_before_claim() -> None:
    artifacts = tuple(
        VerifiedArtifact(f"role-{index:03d}-" + "x" * 230,
                         hashlib.sha256(str(index).encode()).hexdigest(), 1)
        for index in range(100)
    )
    oversized = _forge_verified_source(artifacts)
    service, store, _, sources, spool, adapter = _stack()
    sources.descriptor = oversized
    with pytest.raises(PublicationErrorV1) as error:
        service.publish(PublicationRequest(RUN, "opaque/local-like"))
    assert error.value.code is PublicationCodeV1.SOURCE_INVALID
    assert "secret" not in str(error.value)
    assert store.list("opaque/local-like", 101)[0] == ()
    assert sources.opens == 0 and spool.data == {} and adapter.publish_calls == 0

    class MalformedSource:
        @property
        def payload(self):
            raise RuntimeError("RAW-SECRET-SOURCE-SENTINEL")

    service, store, _, sources, spool, adapter = _stack()
    sources.descriptor = MalformedSource()
    with pytest.raises(PublicationErrorV1) as error:
        service.publish(PublicationRequest(RUN, "opaque/local-like"))
    assert error.value.code is PublicationCodeV1.SOURCE_INVALID
    assert "RAW-SECRET" not in str(error.value)
    assert store.list("opaque/local-like", 101)[0] == ()
    assert sources.opens == 0 and spool.data == {} and adapter.publish_calls == 0


@pytest.mark.parametrize(
    "hostility",
    ["absurd", "float", "unordered", "duplicate", "count", "per", "total"],
)
def test_source_artifact_bounds_fail_before_claim_stream_or_spool(hostility) -> None:
    if hostility == "absurd":
        artifact = VerifiedArtifact("adapter", "1" * 64, 2**63)
        source = _source([(artifact, b"")])
        artifacts = [(artifact, b"")]
    elif hostility == "float":
        artifact = object.__new__(VerifiedArtifact)
        object.__setattr__(artifact, "role", "adapter")
        object.__setattr__(artifact, "sha256", "1" * 64)
        object.__setattr__(artifact, "size_bytes", 1.5)
        source = _forge_verified_source((artifact,))
        object.__setattr__(
            source, "tag",
            AUTH.sign("publication-verified-source/v1", source.payload),
        )
        artifacts = [(artifact, b"")]
    elif hostility == "unordered":
        first = VerifiedArtifact("z", "1" * 64, 1)
        second = VerifiedArtifact("a", "2" * 64, 1)
        source = _forge_verified_source((first, second))
        object.__setattr__(
            source, "tag",
            AUTH.sign("publication-verified-source/v1", source.payload),
        )
        artifacts = [(first, b"z"), (second, b"a")]
    elif hostility == "duplicate":
        first = VerifiedArtifact("a", "1" * 64, 1)
        second = VerifiedArtifact("a", "2" * 64, 1)
        source = _forge_verified_source((first, second))
        object.__setattr__(
            source, "tag",
            AUTH.sign("publication-verified-source/v1", source.payload),
        )
        artifacts = [(first, b"a"), (second, b"b")]
    elif hostility == "count":
        values = tuple(
            VerifiedArtifact(f"r-{index:03d}", hashlib.sha256(
                str(index).encode()).hexdigest(), 1)
            for index in range(101)
        )
        source = _forge_verified_source(values)
        object.__setattr__(
            source, "tag",
            AUTH.sign("publication-verified-source/v1", source.payload),
        )
        artifacts = [(item, b"x") for item in values]
    elif hostility == "per":
        artifact, data = _artifact("adapter", b"12345678901")
        source = _source([(artifact, data)])
        artifacts = [(artifact, data)]
    else:
        first, first_data = _artifact("a", b"123456")
        second, second_data = _artifact("b", b"abcdef")
        source = _source([(first, first_data), (second, second_data)])
        artifacts = [(first, first_data), (second, second_data)]
    stack_artifacts = artifacts
    if hostility in {"unordered", "duplicate", "count"}:
        stack_artifacts = [_artifact()]
    service, store, registry, sources, spool, adapter = _stack(
        artifacts=stack_artifacts
    )
    sources.descriptor = source
    if hostility in {"per", "total"}:
        bounded = _destination(
            maximum_artifact_bytes=10,
            maximum_total_bytes=20 if hostility == "per" else 10,
        )
        registry.entries["opaque/local-like"] = (bounded, adapter)
    with pytest.raises(PublicationErrorV1) as error:
        service.publish(PublicationRequest(RUN, "opaque/local-like"))
    assert error.value.code is PublicationCodeV1.SOURCE_INVALID
    assert store.list("opaque/local-like", 101)[0] == ()
    assert sources.opens == 0 and spool.data == {} and adapter.publish_calls == 0


def test_claim_precedes_spooling_and_failed_spool_never_admits_mutation() -> None:
    artifact, data = _artifact()
    store = StrongInMemoryPublicationStoreV1()

    class ObservingSpool(Spool):
        def open(self, publication_id, role, maximum_bytes):
            assert store.get(publication_id).phase is PublicationPhaseV1.CLAIMED
            raise RuntimeError("local spool unavailable")

    spool = ObservingSpool()
    adapter = Adapter(spool)
    descriptor = _destination()
    service = PublicationOperationsV1(
        store=store,
        destinations=Registry({descriptor.destination_ref: (descriptor, adapter)}),
        sources=Sources([(artifact, data)]), spool=spool, authority=AUTH,
        clock=Clock(),
    )
    with pytest.raises(PublicationErrorV1):
        service.publish(PublicationRequest(RUN, descriptor.destination_ref))
    record = store.list(descriptor.destination_ref, 101)[0][0]
    assert record.phase is PublicationPhaseV1.FAILED_BEFORE_EFFECT
    assert adapter.publish_calls == 0


def test_sequential_and_concurrent_attempts_admit_exactly_one_mutation() -> None:
    service, _, _, _, _, adapter = _stack()
    request = PublicationRequest(RUN, "opaque/local-like")
    assert service.publish(request).state is PublicationState.VERIFIED
    assert service.publish(request).state is PublicationState.VERIFIED
    assert adapter.publish_calls == 1

    barrier = Barrier(2)
    service, _, _, _, _, adapter = _stack(barrier=barrier)
    results = []
    threads = [Thread(target=lambda: results.append(service.publish(request))) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=5)
    assert len(results) == 2
    assert adapter.publish_calls == 1
    assert {item.state for item in results} <= {
        PublicationState.CLAIMED, PublicationState.TRANSFERRING,
        PublicationState.COMMITTED, PublicationState.VERIFIED,
    }


def test_lost_publish_return_is_ambiguous_then_lookup_only_recovery() -> None:
    service, store, _, _, _, adapter = _stack(mode="raise_after_effect")
    request = PublicationRequest(RUN, "opaque/local-like")
    first = service.publish(request)
    assert first.state is PublicationState.AMBIGUOUS
    assert adapter.publish_calls == 1
    second = service.publish(request)
    assert second.state is PublicationState.VERIFIED
    assert adapter.publish_calls == 1
    assert adapter.lookup_calls == 1
    assert store.get(second.publication.publication_id).phase is PublicationPhaseV1.VERIFIED


@pytest.mark.parametrize(("failed_complete", "lookups"), [(1, 1), (2, 0)])
def test_lost_commit_or_final_return_replays_durable_state_without_republish(
        failed_complete, lookups) -> None:
    class LostCompleteStore(StrongInMemoryPublicationStoreV1):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def complete_transfer(self, ownership, receipt, verified, timestamp):
            self.calls += 1
            result = super().complete_transfer(
                ownership, receipt, verified, timestamp
            )
            if self.calls == failed_complete:
                raise RuntimeError("lost durable return with secret context")
            return result

    store = LostCompleteStore()
    service, store, _, _, _, adapter = _stack(store=store)
    result = service.publish(PublicationRequest(RUN, "opaque/local-like"))
    assert result.state is PublicationState.VERIFIED
    assert adapter.publish_calls == 1
    assert adapter.lookup_calls == lookups


def test_recovering_an_orphaned_transfer_is_lookup_only() -> None:
    service, store, registry, _, _, adapter = _stack()
    destination = registry.entries["opaque/local-like"][0]
    artifact, data = _artifact()
    source = _source([(artifact, data)])
    command = PublicationCommandV1.build(
        run=RUN, source_identity_digest=source.source_identity_digest,
        source_inventory=source.artifacts,
        destination_ref=destination.destination_ref,
        destination_identity_digest=destination.identity_digest,
        destination_configuration_digest=destination.configuration_digest,
        destination_policy_digest=destination.policy_digest,
        maximum_artifact_bytes=destination.maximum_artifact_bytes,
        maximum_total_bytes=destination.maximum_total_bytes,
        destination_authority_ref=destination.authority_ref,
        destination_key_ref=destination.key_ref,
    )
    claim = PublicationRecordV1.claim(command, "2026-08-27T12:00:00Z")
    store.claim(claim)
    admission = store.begin_transfer(
        command.publication_id, claim.record_digest, "2026-08-27T12:00:01Z"
    )
    admitted = admission.record
    store.mark_orphaned(admission.ownership)
    checked = service.verify(PublicationRef(command.publication_id,
                                            command.destination_ref))
    assert not checked.verified
    assert store.get(command.publication_id).phase is PublicationPhaseV1.AMBIGUOUS
    assert adapter.publish_calls == 0
    assert adapter.lookup_calls == 1


@pytest.mark.parametrize(
    ("mode", "phase"),
    [
        ("absent", PublicationPhaseV1.ABSENT),
        ("indeterminate", PublicationPhaseV1.AMBIGUOUS),
        ("lookup_raise", PublicationPhaseV1.AMBIGUOUS),
        ("conflict", PublicationPhaseV1.CONFLICT),
    ],
)
def test_lookup_outcomes_are_closed_terminal_and_never_retry(mode, phase) -> None:
    service, store, _, _, _, adapter = _stack(mode="raise_after_effect")
    result = service.publish(PublicationRequest(RUN, "opaque/local-like"))
    adapter.mode = mode
    recovered = service.publish(PublicationRequest(RUN, "opaque/local-like"))
    record = store.get(result.publication.publication_id)
    assert record.phase is phase
    assert adapter.publish_calls == 1
    calls = adapter.lookup_calls
    service.publish(PublicationRequest(RUN, "opaque/local-like"))
    assert adapter.publish_calls == 1
    if phase is not PublicationPhaseV1.AMBIGUOUS:
        assert adapter.lookup_calls == calls


@pytest.mark.parametrize("mode", ["tamper", "truncate", "trailing"])
def test_readback_bytes_are_accepted_only_at_exact_eof(mode) -> None:
    service, store, _, _, _, adapter = _stack(mode=mode)
    result = service.publish(PublicationRequest(RUN, "opaque/local-like"))
    assert result.state is PublicationState.AMBIGUOUS
    assert store.get(result.publication.publication_id).phase is PublicationPhaseV1.AMBIGUOUS
    assert adapter.publish_calls == 1


def test_malformed_cross_bound_receipt_and_unauthenticated_lookup_fail_closed() -> None:
    service, store, _, _, _, adapter = _stack(mode="bad_receipt")
    result = service.publish(PublicationRequest(RUN, "opaque/local-like"))
    assert result.state is PublicationState.AMBIGUOUS
    assert adapter.publish_calls == 1


@pytest.mark.parametrize(
    "change",
    [
        {"publication_id": "1" * 64},
        {"command_digest": "2" * 64},
        {"run": TrainingRunRef("cross-run", "project-1")},
        {"source_identity_digest": "3" * 64},
        {"destination_ref": "opaque/cross-destination"},
        {"destination_identity_digest": "4" * 64},
        {"mutation_id": "5" * 64},
        {"claim_digest": "6" * 64},
    ],
)
def test_authenticated_cross_bound_found_receipts_terminalize_conflict(change) -> None:
    service, store, _, _, _, adapter = _stack(mode="raise_after_effect")
    result = service.publish(PublicationRequest(RUN, "opaque/local-like"))
    changed = replace(adapter.receipt, **change, tag=ZERO)
    adapter.receipt = replace(
        changed, tag=AUTH.sign("publication-receipt/v1", changed.payload)
    )
    adapter.mode = "success"
    service.publish(PublicationRequest(RUN, "opaque/local-like"))
    assert store.get(result.publication.publication_id).phase is PublicationPhaseV1.CONFLICT
    assert adapter.publish_calls == 1


def test_inventory_is_exact_canonical_bounded_and_cross_checked() -> None:
    first = DestinationArtifactV1("a", "bundle/a", "1" * 64, 1)
    second = DestinationArtifactV1("b", "bundle/b", "2" * 64, 2)
    assert DestinationInventoryV1((first, second)).artifacts == (first, second)
    with pytest.raises(ValueError):
        DestinationInventoryV1((second, first))
    with pytest.raises(ValueError):
        DestinationInventoryV1((first, first))
    with pytest.raises(ValueError):
        DestinationArtifactV1("a", "bundle/a", "1" * 64, 2**63)


@pytest.mark.parametrize("change", ["missing", "extra", "renamed", "resized"])
def test_authenticated_wrong_inventory_terminalizes_conflict(change) -> None:
    artifacts = [_artifact("a", b"a"), _artifact("b", b"bb")]
    service, store, _, _, _, adapter = _stack(
        mode="raise_after_effect", artifacts=artifacts
    )
    result = service.publish(PublicationRequest(RUN, "opaque/local-like"))
    items = adapter.receipt.inventory.inventory.artifacts
    if change == "missing":
        values = items[:1]
    elif change == "extra":
        values = items + (DestinationArtifactV1(
            "c", "bundle/c", "3" * 64, 3
        ),)
    elif change == "renamed":
        values = (replace(items[0], role="aa"), items[1])
    else:
        values = (replace(items[0], size_bytes=2), items[1])
    changed = replace(
        adapter.receipt.inventory, inventory=DestinationInventoryV1(values), tag=ZERO
    )
    changed = replace(
        changed, tag=AUTH.sign("publication-destination-inventory/v1",
                               changed.payload)
    )
    receipt = replace(adapter.receipt, inventory=changed, tag=ZERO)
    adapter.receipt = replace(
        receipt, tag=AUTH.sign("publication-receipt/v1", receipt.payload)
    )
    adapter.mode = "success"
    service.publish(PublicationRequest(RUN, "opaque/local-like"))
    assert store.get(result.publication.publication_id).phase is PublicationPhaseV1.CONFLICT
    assert adapter.publish_calls == 1
    adapter.mode = "indeterminate"
    original = adapter.lookup
    adapter.lookup = lambda command, claim: replace(original(command, claim), tag="f" * 64)
    assert service.publish(PublicationRequest(RUN, "opaque/local-like")).state is PublicationState.AMBIGUOUS
    assert adapter.publish_calls == 1


def test_store_only_publication_listing_and_strict_complete_page_bounds() -> None:
    service, store, registry, _, _, adapter = _stack()
    result = service.publish(PublicationRequest(RUN, "opaque/local-like"))
    before = (registry.resolve_calls, adapter.publish_calls, adapter.lookup_calls,
              adapter.read_calls)
    page = service.publications("opaque/local-like")
    assert page.publications == (result,)
    assert (registry.resolve_calls, adapter.publish_calls, adapter.lookup_calls,
            adapter.read_calls) == before

    class OverflowStore:
        def list(self, destination_ref, limit):
            return tuple(store.get(result.publication.publication_id) for _ in range(101)), False
    overflow = PublicationOperationsV1(
        store=OverflowStore(), destinations=registry, sources=Sources([_artifact()]),
        spool=Spool(), authority=AUTH, clock=Clock(),
    )
    with pytest.raises(PublicationErrorV1) as error:
        overflow.publications("opaque/local-like")
    assert error.value.code is PublicationCodeV1.PAGE_INCOMPLETE


@pytest.mark.parametrize("hostility", ["malformed", "cross", "duplicate", "reordered"])
def test_publication_listing_rejects_hostile_complete_pages(hostility) -> None:
    service, store, registry, sources, spool, adapter = _stack()
    result = service.publish(PublicationRequest(RUN, "opaque/local-like"))
    first = store.get(result.publication.publication_id)
    descriptor = registry.entries["opaque/local-like"][0]
    artifact = first.command.source_inventory[0]
    second_command = PublicationCommandV1.build(
        run=TrainingRunRef("run-2", "project-1"),
        source_identity_digest="b" * 64,
        source_inventory=(artifact,),
        destination_ref=descriptor.destination_ref,
        destination_identity_digest=descriptor.identity_digest,
        destination_configuration_digest=descriptor.configuration_digest,
        destination_policy_digest=descriptor.policy_digest,
        maximum_artifact_bytes=descriptor.maximum_artifact_bytes,
        maximum_total_bytes=descriptor.maximum_total_bytes,
        destination_authority_ref=descriptor.authority_ref,
        destination_key_ref=descriptor.key_ref,
    )
    second = PublicationRecordV1.claim(
        second_command, "2026-08-27T12:30:00Z"
    )
    cross_descriptor = _destination("opaque/cross")
    cross_command = PublicationCommandV1.build(
        run=RUN, source_identity_digest="c" * 64,
        source_inventory=(artifact,),
        destination_ref=cross_descriptor.destination_ref,
        destination_identity_digest=cross_descriptor.identity_digest,
        destination_configuration_digest=cross_descriptor.configuration_digest,
        destination_policy_digest=cross_descriptor.policy_digest,
        maximum_artifact_bytes=cross_descriptor.maximum_artifact_bytes,
        maximum_total_bytes=cross_descriptor.maximum_total_bytes,
        destination_authority_ref=cross_descriptor.authority_ref,
        destination_key_ref=cross_descriptor.key_ref,
    )
    cross = PublicationRecordV1.claim(cross_command, "2026-08-27T12:31:00Z")
    ordered = tuple(sorted((first, second),
                           key=lambda item: item.command.publication_id))
    values = {
        "malformed": (object(),),
        "cross": (cross,),
        "duplicate": (first, first),
        "reordered": tuple(reversed(ordered)),
    }[hostility]

    class HostileStore:
        def list(self, destination_ref, limit):
            return values, True

    hostile = PublicationOperationsV1(
        store=HostileStore(), destinations=registry, sources=sources,
        spool=spool, authority=AUTH, clock=Clock(),
    )
    before = (adapter.publish_calls, adapter.lookup_calls, adapter.read_calls)
    with pytest.raises(PublicationErrorV1) as error:
        hostile.publications("opaque/local-like")
    assert error.value.code is PublicationCodeV1.PAGE_INCOMPLETE
    assert (adapter.publish_calls, adapter.lookup_calls, adapter.read_calls) == before


def test_destinations_are_authenticated_complete_and_bounded() -> None:
    service, _, registry, _, _, _ = _stack()
    assert service.destinations().destinations[0].display_name == "Target local-like"
    descriptor, adapter = registry.entries["opaque/local-like"]
    registry.entries["opaque/local-like"] = (replace(descriptor, tag="f" * 64), adapter)
    with pytest.raises(PublicationErrorV1) as error:
        service.destinations()
    assert error.value.code is PublicationCodeV1.DESTINATION_INVALID


def test_verify_cannot_orphan_or_lookup_while_transfer_owner_is_live() -> None:
    service, store, _, _, _, adapter = _stack()
    adapter.entered, adapter.release = Event(), Event()
    results = []
    thread = Thread(target=lambda: results.append(
        service.publish(PublicationRequest(RUN, "opaque/local-like"))))
    thread.start()
    assert adapter.entered.wait(timeout=5)
    record = store.list("opaque/local-like", 101)[0][0]
    assert record.phase is PublicationPhaseV1.TRANSFERRING
    checked = service.verify(PublicationRef(
        record.command.publication_id, record.command.destination_ref
    ))
    assert not checked.verified
    assert store.get(record.command.publication_id).phase is PublicationPhaseV1.TRANSFERRING
    assert adapter.lookup_calls == 0
    adapter.release.set()
    thread.join(timeout=5)
    assert results[0].state is PublicationState.VERIFIED
    assert adapter.publish_calls == 1


def test_orphan_fencing_issues_one_lookup_permit_and_stale_owner_cannot_commit() -> None:
    service, store, _, _, _, adapter = _stack(mode="raise_after_effect")
    request = PublicationRequest(RUN, "opaque/local-like")
    ambiguous = service.publish(request)
    record = store.get(ambiguous.publication.publication_id)
    ownership = record.ownership_history[-1]
    permit = record.recovery_permits[-1]
    assert permit.fenced_ownership_id == ownership.ownership_id
    adapter.mode = "absent"
    service.publish(request)
    terminal = store.get(ambiguous.publication.publication_id)
    assert terminal.phase is PublicationPhaseV1.ABSENT
    assert terminal.tombstone.recovery_permit_id == permit.permit_id
    with pytest.raises(PublicationErrorV1) as error:
        store.complete_transfer(ownership, adapter.receipt, False,
                                "2026-08-27T12:01:00Z")
    assert error.value.code is PublicationCodeV1.STATE_CONFLICT
    assert adapter.publish_calls == 1


@pytest.mark.parametrize(
    "field",
    [
        "claim_digest", "destination_ref", "destination_configuration_digest",
        "destination_policy_digest", "fenced_ownership_id",
        "recovery_permit_id", "mutation_registry_digest", "authority_ref",
        "key_ref",
    ],
)
def test_wrong_tombstone_bindings_or_alternate_valid_signers_never_prove_absence(field) -> None:
    service, store, _, _, _, adapter = _stack(mode="raise_after_effect")
    request = PublicationRequest(RUN, "opaque/local-like")
    ambiguous = service.publish(request)
    adapter.mode = "absent"
    original = adapter.lookup

    def changed_lookup(command, permit):
        evidence = original(command, permit)
        value = "alternate-key" if field == "key_ref" else (
            "alternate-authority" if field == "authority_ref" else "7" * 64
        )
        tombstone = replace(evidence.tombstone, **{field: value}, tag=ZERO)
        tombstone = replace(
            tombstone,
            tag=AUTH.sign("publication-tombstone/v1", tombstone.payload,
                          tombstone.key_ref),
        )
        changed = replace(evidence, tombstone=tombstone, tag=ZERO)
        return replace(
            changed,
            tag=AUTH.sign("publication-lookup/v1", changed.payload,
                          changed.key_ref),
        )

    adapter.lookup = changed_lookup
    service.publish(request)
    record = store.get(ambiguous.publication.publication_id)
    assert record.phase is not PublicationPhaseV1.ABSENT
    assert adapter.publish_calls == 1


def test_alternate_valid_receipt_and_inventory_signers_fail_closed() -> None:
    class AlternateSignerAdapter(Adapter):
        def publish_once(self, command, source, ownership):
            receipt = super().publish_once(command, source, ownership)
            inventory = replace(
                receipt.inventory, authority_ref="alternate-authority",
                key_ref="alternate-key", tag=ZERO,
            )
            inventory = replace(
                inventory,
                tag=AUTH.sign("publication-destination-inventory/v1",
                              inventory.payload, inventory.key_ref),
            )
            changed = replace(
                receipt, inventory=inventory,
                authority_ref="alternate-authority", key_ref="alternate-key",
                tag=ZERO,
            )
            return replace(
                changed, tag=AUTH.sign("publication-receipt/v1",
                                       changed.payload, changed.key_ref)
            )

    service, store, registry, sources, spool, _ = _stack()
    adapter = AlternateSignerAdapter(spool)
    descriptor = registry.entries["opaque/local-like"][0]
    registry.entries["opaque/local-like"] = (descriptor, adapter)
    result = service.publish(PublicationRequest(RUN, "opaque/local-like"))
    assert result.state is PublicationState.AMBIGUOUS
    assert adapter.publish_calls == 1
    assert store.get(result.publication.publication_id).phase is PublicationPhaseV1.AMBIGUOUS


def test_alternate_valid_lookup_signer_is_indeterminate() -> None:
    service, store, _, _, _, adapter = _stack(mode="raise_after_effect")
    request = PublicationRequest(RUN, "opaque/local-like")
    result = service.publish(request)
    adapter.mode = "absent"
    original = adapter.lookup

    def alternate(command, permit):
        evidence = original(command, permit)
        changed = replace(
            evidence, authority_ref="alternate-authority",
            key_ref="alternate-key", tag=ZERO,
        )
        return replace(
            changed, tag=AUTH.sign("publication-lookup/v1", changed.payload,
                                   changed.key_ref)
        )

    adapter.lookup = alternate
    service.publish(request)
    assert store.get(result.publication.publication_id).phase is PublicationPhaseV1.AMBIGUOUS
    assert adapter.publish_calls == 1


def test_generic_cas_cannot_commit_forged_receipt_or_replace_history() -> None:
    service, store, _, _, _, adapter = _stack()
    result = service.publish(PublicationRequest(RUN, "opaque/local-like"))
    verified = store.get(result.publication.publication_id)
    event = verified.history[-1]
    with pytest.raises(ValueError):
        replace(event, timestamp="2026-08-27T13:00:00Z")

    artifact, _ = _artifact()
    destination = _destination()
    source = _source([(artifact, b"adapter")])
    command = PublicationCommandV1.build(
        run=RUN, source_identity_digest=source.source_identity_digest,
        source_inventory=source.artifacts,
        destination_ref=destination.destination_ref,
        destination_identity_digest=destination.identity_digest,
        destination_configuration_digest=destination.configuration_digest,
        destination_policy_digest=destination.policy_digest,
        maximum_artifact_bytes=destination.maximum_artifact_bytes,
        maximum_total_bytes=destination.maximum_total_bytes,
        destination_authority_ref=destination.authority_ref,
        destination_key_ref=destination.key_ref,
    )
    store = StrongInMemoryPublicationStoreV1()
    claim = PublicationRecordV1.claim(command, "2026-08-27T12:00:00Z")
    store.claim(claim)
    admission = store.begin_transfer(
        command.publication_id, claim.record_digest, "2026-08-27T12:00:01Z"
    )
    inventory = DestinationInventoryV1((DestinationArtifactV1(
        artifact.role, "bundle/adapter", artifact.sha256, artifact.size_bytes
    ),))
    receipt = _receipt(command, admission.ownership, inventory)
    forged = admission.record._advance(
        PublicationPhaseV1.COMMITTED, "2026-08-27T13:00:00Z",
        receipt=receipt,
    )
    with pytest.raises(PublicationErrorV1) as error:
        store.compare_and_swap(admission.record.record_digest, forged)
    assert error.value.code is PublicationCodeV1.STATE_CONFLICT


def test_throwing_store_boundaries_are_closed_and_never_recreate_mutation() -> None:
    class ThrowCompleteStore(StrongInMemoryPublicationStoreV1):
        def complete_transfer(self, ownership, receipt, verified, timestamp):
            raise RuntimeError("database password and raw traceback")

    store = ThrowCompleteStore()
    service, store, _, _, _, adapter = _stack(store=store)
    first = service.publish(PublicationRequest(RUN, "opaque/local-like"))
    assert first.state is PublicationState.VERIFIED
    assert adapter.lookup_calls == 1
    assert "password" not in str(first)
    service.publish(PublicationRequest(RUN, "opaque/local-like"))
    assert adapter.publish_calls == 1

    class ThrowGetStore(StrongInMemoryPublicationStoreV1):
        def get(self, publication_id):
            raise RuntimeError("database password and raw traceback")

    service, _, _, _, _, adapter = _stack(store=ThrowGetStore())
    with pytest.raises(PublicationErrorV1) as error:
        service.publish(PublicationRequest(RUN, "opaque/local-like"))
    assert error.value.code is PublicationCodeV1.STATE_CONFLICT
    assert "password" not in str(error.value)
    assert adapter.publish_calls == 0


def test_each_remaining_throwing_store_boundary_fails_closed() -> None:
    class ThrowClaim(StrongInMemoryPublicationStoreV1):
        def claim(self, record):
            raise RuntimeError("secret claim traceback")

    service, _, _, _, _, adapter = _stack(store=ThrowClaim())
    with pytest.raises(PublicationErrorV1) as error:
        service.publish(PublicationRequest(RUN, "opaque/local-like"))
    assert error.value.code is PublicationCodeV1.STATE_CONFLICT
    assert adapter.publish_calls == 0

    class ThrowBegin(StrongInMemoryPublicationStoreV1):
        def begin_transfer(self, publication_id, expected_record_digest, timestamp):
            raise RuntimeError("secret begin traceback")

    service, store, _, _, _, adapter = _stack(store=ThrowBegin())
    assert service.publish(PublicationRequest(
        RUN, "opaque/local-like")).state is PublicationState.CLAIMED
    assert adapter.publish_calls == 0

    class ThrowCAS(StrongInMemoryPublicationStoreV1):
        def compare_and_swap(self, expected_record_digest, descendant):
            raise RuntimeError("secret cas traceback")

    service, store, _, _, _, adapter = _stack(
        store=ThrowCAS(),
        stream_factory=lambda run, item, raw: Stream(
            run, item, raw, chunks=(raw[:-1],)
        ),
    )
    with pytest.raises(PublicationErrorV1) as error:
        service.publish(PublicationRequest(RUN, "opaque/local-like"))
    assert error.value.code is PublicationCodeV1.SOURCE_CONTENT_INVALID
    assert adapter.publish_calls == 0

    class ThrowRelinquish(StrongInMemoryPublicationStoreV1):
        def relinquish_uncertain(self, ownership, timestamp):
            raise RuntimeError("secret relinquish traceback")

    service, store, _, _, _, adapter = _stack(
        store=ThrowRelinquish(), mode="raise_after_effect"
    )
    first = service.publish(PublicationRequest(RUN, "opaque/local-like"))
    assert first.state is PublicationState.TRANSFERRING
    service.publish(PublicationRequest(RUN, "opaque/local-like"))
    assert adapter.publish_calls == 1

    class ThrowRecover(StrongInMemoryPublicationStoreV1):
        def recover_transfer(self, publication_id, command_digest, timestamp):
            raise RuntimeError("secret recover traceback")

    service, store, _, _, _, adapter = _stack(
        store=ThrowRecover(), mode="raise_after_effect"
    )
    first = service.publish(PublicationRequest(RUN, "opaque/local-like"))
    assert first.state is PublicationState.AMBIGUOUS
    service.publish(PublicationRequest(RUN, "opaque/local-like"))
    assert adapter.publish_calls == 1
    assert adapter.lookup_calls == 0


@pytest.mark.parametrize(
    "change",
    [
        {"display_name": "Rebound identity"},
        {"configuration_digest": "8" * 64},
        {"policy_digest": "9" * 64},
        {"authority_ref": "alternate-authority"},
        {"key_ref": "alternate-key"},
    ],
)
def test_rebound_destination_descriptor_cannot_recover_existing_publication(change) -> None:
    service, store, registry, _, _, adapter1 = _stack(mode="raise_after_effect")
    result = service.publish(PublicationRequest(RUN, "opaque/local-like"))
    assert store.get(result.publication.publication_id).phase is PublicationPhaseV1.AMBIGUOUS
    descriptor1 = registry.entries["opaque/local-like"][0]
    descriptor2 = replace(descriptor1, **change, tag=ZERO)
    descriptor2 = replace(
        descriptor2,
        tag=AUTH.sign("publication-destination/v1", descriptor2.payload,
                      descriptor2.key_ref),
    )
    adapter2 = Adapter(Spool(), mode="absent")
    registry.entries["opaque/local-like"] = (descriptor2, adapter2)
    checked = service.verify(result.publication)
    assert not checked.verified
    assert store.get(result.publication.publication_id).phase is PublicationPhaseV1.AMBIGUOUS
    assert adapter2.publish_calls == adapter2.lookup_calls == adapter2.read_calls == 0
    assert adapter1.publish_calls == 1


def test_exact_unchanged_descriptor_allows_lookup_recovery() -> None:
    service, store, registry, _, _, adapter = _stack(mode="raise_after_effect")
    result = service.publish(PublicationRequest(RUN, "opaque/local-like"))
    descriptor = registry.entries["opaque/local-like"][0]
    record = store.get(result.publication.publication_id)
    assert service._descriptor_matches_command(descriptor, record.command)
    adapter.mode = "success"
    assert service.verify(result.publication).verified
    assert adapter.lookup_calls == 1
    assert adapter.publish_calls == 1


def _persist_claim_before_admission(service, store, registry, sources):
    descriptor = registry.entries["opaque/local-like"][0]
    source = sources.describe(RUN)
    command = PublicationCommandV1.build(
        run=RUN, source_identity_digest=source.source_identity_digest,
        source_inventory=source.artifacts,
        destination_ref=descriptor.destination_ref,
        destination_identity_digest=descriptor.identity_digest,
        destination_configuration_digest=descriptor.configuration_digest,
        destination_policy_digest=descriptor.policy_digest,
        maximum_artifact_bytes=descriptor.maximum_artifact_bytes,
        maximum_total_bytes=descriptor.maximum_total_bytes,
        destination_authority_ref=descriptor.authority_ref,
        destination_key_ref=descriptor.key_ref,
    )
    record = PublicationRecordV1.claim(command, "2026-08-27T11:59:59Z")
    assert store.claim(record) == (record, True)
    return record


def test_claimed_replay_resumes_spooling_and_single_transfer_admission() -> None:
    service, store, registry, sources, _, adapter = _stack()
    claimed = _persist_claim_before_admission(
        service, store, registry, sources
    )
    result = service.publish(PublicationRequest(RUN, "opaque/local-like"))
    assert result.state is PublicationState.VERIFIED
    record = store.get(claimed.command.publication_id)
    assert record.phase is PublicationPhaseV1.VERIFIED
    assert len(record.ownership_history) == 1
    assert adapter.publish_calls == 1


def test_concurrent_claimed_replay_still_has_one_owner_and_mutation() -> None:
    service, store, registry, sources, _, adapter = _stack()
    claimed = _persist_claim_before_admission(
        service, store, registry, sources
    )
    request = PublicationRequest(RUN, "opaque/local-like")
    results = []
    threads = [Thread(target=lambda: results.append(service.publish(request)))
               for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert len(results) == 2
    record = store.get(claimed.command.publication_id)
    assert len(record.ownership_history) == 1
    assert adapter.publish_calls == 1
    assert any(item.state is PublicationState.VERIFIED for item in results)


def test_protocol_shape_training_independence_and_forbidden_boundaries() -> None:
    assert set(name for name, member in ArtifactsOperations.__dict__.items()
               if not name.startswith("_") and inspect.isfunction(member)) == {
        "destinations", "publications", "publish", "verify",
    }
    assert set(name for name, member in PublicationOperationsV1.__dict__.items()
               if not name.startswith("_") and callable(member)) == {
        "destinations", "publications", "publish", "verify",
    }
    source = inspect.getsource(__import__(
        "tuner.execution.coordinator_v1.publication", fromlist=["publication"]
    )).lower()
    forbidden = (
        "sqlite", "huggingface_hub", "modal.", "runpod", "boto", "requests",
        "foundationeffect", "effectkind", "legacy", "credential", "hf_token",
    )
    assert all(token not in source for token in forbidden)
    assert "trainingrunstate" not in source
