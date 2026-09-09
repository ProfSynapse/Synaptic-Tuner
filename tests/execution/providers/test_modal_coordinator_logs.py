"""Contracts for generic Foundation-native Modal log chunks."""

import hashlib
import json
from dataclasses import replace

import pytest

from synaptic_tuner.api.v1.runs_facade import RunLogEntry, RunLogLevel
from tuner.execution.providers.modal.contracts import BoundsPolicyV1, canonical_json
from tuner.execution.providers.modal.coordinator_logs import (
    ModalCoordinatorLogChunk, validate_modal_log_chain,
)


def entry(sequence=0, *, timestamp="2026-09-09T12:00:00Z", message="ready"):
    return RunLogEntry(
        sequence, timestamp, RunLogLevel.INFO, "training_progress", message,
        len(message.encode("utf-8")),
    )


def chunk(sequence=0, *, previous="0" * 64, records=None, job_ref="job-a"):
    records = tuple(records or (entry(0),))
    payload = canonical_json([record.to_dict() for record in records])
    return ModalCoordinatorLogChunk(
        1, sequence, previous, hashlib.sha256(payload).hexdigest(), job_ref,
        "submit-effect", "a" * 64, "submit-nonce", records,
    )


def test_canonical_roundtrip_retains_exact_generic_entries():
    original = chunk(records=(entry(0), entry(1, message="step two")))
    parsed = ModalCoordinatorLogChunk.parse(original.canonical_bytes)
    assert parsed == original
    assert parsed.records == original.records
    assert parsed.records is not original.records


def test_invalid_rfc3339_timestamp_is_rejected_by_generic_parser():
    document = json.loads(chunk().canonical_bytes)
    document["records"][0]["timestamp"] = "yesterday"
    document["payload_digest"] = hashlib.sha256(
        canonical_json(document["records"])
    ).hexdigest()
    with pytest.raises(ValueError):
        ModalCoordinatorLogChunk.parse(canonical_json(document))


def test_changed_payload_digest_is_rejected():
    document = json.loads(chunk().canonical_bytes)
    document["payload_digest"] = "b" * 64
    with pytest.raises(ValueError):
        ModalCoordinatorLogChunk.parse(canonical_json(document))


def test_chain_requires_contiguous_chunk_and_global_record_sequences():
    first = chunk(records=(entry(0), entry(1)))
    second = chunk(1, previous=first.chunk_digest, records=(entry(2),))
    assert validate_modal_log_chain((first, second)) == second.chunk_digest
    with pytest.raises(ValueError, match="record sequence"):
        validate_modal_log_chain((first, replace(second, records=(entry(3),), payload_digest=hashlib.sha256(canonical_json([entry(3).to_dict()])).hexdigest())))
    with pytest.raises(ValueError, match="chain mismatch"):
        validate_modal_log_chain((first, replace(second, sequence=2)))


def test_chain_rejects_changed_identity_and_predecessor():
    first = chunk()
    second = chunk(1, previous=first.chunk_digest, records=(entry(1),))
    with pytest.raises(ValueError, match="chain mismatch"):
        validate_modal_log_chain((first, replace(second, job_ref="job-b")))
    with pytest.raises(ValueError, match="chain mismatch"):
        validate_modal_log_chain((first, replace(second, previous_digest="c" * 64)))


def test_empty_chunks_records_and_nonexact_container_are_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        validate_modal_log_chain(())
    with pytest.raises(ValueError, match="exact tuple"):
        validate_modal_log_chain([chunk()])
    with pytest.raises(ValueError, match="nonempty"):
        replace(chunk(), records=(), payload_digest=hashlib.sha256(b"[]").hexdigest())


@pytest.mark.parametrize("field,value", [("generation", True), ("sequence", True)])
def test_boolean_integer_fields_are_rejected(field, value):
    values = dict(
        generation=1, sequence=0, previous_digest="0" * 64,
        payload_digest=hashlib.sha256(canonical_json([entry().to_dict()])).hexdigest(),
        job_ref="job-a", effect_id="submit-effect", plan_digest="a" * 64,
        invocation_nonce="submit-nonce", records=(entry(),),
    )
    values[field] = value
    with pytest.raises(ValueError):
        ModalCoordinatorLogChunk(**values)


def test_parse_and_chain_enforce_record_bounds():
    original = chunk(records=(entry(0), entry(1)))
    bounds = replace(BoundsPolicyV1(), max_log_records=1)
    with pytest.raises(ValueError):
        ModalCoordinatorLogChunk.parse(original.canonical_bytes, bounds=bounds)
    first = chunk(records=(entry(0),))
    second = chunk(1, previous=first.chunk_digest, records=(entry(1),))
    with pytest.raises(ValueError):
        validate_modal_log_chain((first, second), bounds=bounds)


def test_parse_rejects_unknown_fields_and_noncanonical_bytes():
    document = json.loads(chunk().canonical_bytes)
    document["credential"] = "secret"
    with pytest.raises(ValueError):
        ModalCoordinatorLogChunk.parse(canonical_json(document))
    with pytest.raises(ValueError):
        ModalCoordinatorLogChunk.parse(b'{ "schema": "synaptic.modal-log-chunk/v2" }')


def test_parse_rejects_bytes_subclass():
    class Bytes(bytes):
        pass
    with pytest.raises(TypeError, match="exact immutable"):
        ModalCoordinatorLogChunk.parse(Bytes(chunk().canonical_bytes))


def test_chain_reconstructs_and_rejects_mutated_exact_chunk():
    value = chunk()
    object.__setattr__(value, "payload_digest", "f" * 64)
    with pytest.raises(ValueError):
        validate_modal_log_chain((value,))
