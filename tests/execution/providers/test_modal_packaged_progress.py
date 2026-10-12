"""Signed packaged progress through create-only files and the real SDK facade."""

from __future__ import annotations

import hashlib
import hmac
import sys
from dataclasses import replace
from threading import Event

import pytest

from synaptic_tuner.api.v1.observations import (
    OBSERVATION_SCHEMA_VERSION, ObservationFamily, ObservationKind,
    ObservationRecordV1, ObservationStreamRef,
    TrainingMetricObservedPayloadV1, TrainingPhaseObservedPayloadV1,
)
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1._contract import canonical_bytes
from tuner.execution.observation_logs import (
    decode_training_progress_log, encode_training_progress_log,
)

from tuner.execution.providers.modal import mounted_io
from tuner.execution.providers.modal.packaged_progress import (
    PackagedProgressPublisher, PROGRESS_PURPOSE, ZERO_DIGEST,
    make_progress_file, parse_progress_file, progress_path,
)
from tests.execution.providers.test_modal_packaged_reader import _reader
from tests import test_modal_volume_root_binding as root_tests


@pytest.fixture
def volume_layout(monkeypatch):
    yield from root_tests.layout.__wrapped__(monkeypatch)


class Auth:
    def sign(self, purpose, payload, key_ref):
        assert purpose == PROGRESS_PURPOSE and key_ref == "evidence-key"
        return hmac.new(b"progress-test-key", payload, hashlib.sha256).digest()

    def verify(self, purpose, payload, tag, key_ref):
        return hmac.compare_digest(self.sign(purpose, payload, key_ref), tag)


def metric(step):
    return TrainingMetricObservedPayloadV1(
        step=step, total_steps=4, loss=0.25, learning_rate=0.00015,
        epoch=0.25, train_elapsed_seconds=2.5, job_elapsed_seconds=None,
        gpu_current_allocated_bytes=None, gpu_current_reserved_bytes=None,
        gpu_peak_allocated_bytes=None, gpu_peak_reserved_bytes=None,
        gpu_utilization_percent=None, tokens_per_second=12.5,
        estimated_samples_per_second=None,
    )


def _identity(binding):
    preparation = binding.command.preparation
    return dict(
        signer=Auth(), key_ref="evidence-key",
        effect_id=binding.command.operation.effect.effect_id,
        command_digest=binding.command_digest,
        provider_job_ref="fc-1",
        execution_binding_digest=binding.execution_binding.binding_digest,
        run=TrainingRunRef(preparation.run_id, preparation.project_ref),
        max_records=3,
    )


def test_signed_envelope_binds_run_sequence_previous_digest_and_exact_metric():
    binding, _, _, _, _, _ = _reader()
    identity = _identity(binding)
    record = ObservationRecordV1(
        OBSERVATION_SCHEMA_VERSION,
        ObservationStreamRef(ObservationFamily.TRAINING,
                             identity["run"].project_ref, identity["run"].run_id),
        0, "2026-10-07T10:15:30Z",
        ObservationKind.TRAINING_METRIC_OBSERVED, metric(1),
    )
    raw, digest = make_progress_file(
        **identity, index=0, previous_digest=ZERO_DIGEST, record=record,
    )
    entry, parsed_digest, maximum, exhausted = parse_progress_file(
        raw, verifier=Auth(), key_ref="evidence-key",
        effect_id=identity["effect_id"],
        command_digest=identity["command_digest"],
        provider_job_ref="fc-1",
        execution_binding_digest=identity["execution_binding_digest"],
        run=identity["run"], max_records=None,
        index=0, previous_digest=ZERO_DIGEST,
    )
    assert parsed_digest == digest and maximum == 3 and exhausted is False
    assert entry.sequence == 0 and '"learning_rate":0.00015' in entry.message
    with pytest.raises(ValueError, match="binding mismatch"):
        parse_progress_file(
            raw, verifier=Auth(), key_ref="evidence-key",
            effect_id=identity["effect_id"],
            command_digest=identity["command_digest"],
            provider_job_ref="fc-other",
            execution_binding_digest=identity["execution_binding_digest"],
            run=identity["run"], max_records=3,
            index=0, previous_digest=ZERO_DIGEST,
        )


def test_early_signed_budget_status_is_rejected_at_write_and_read():
    binding, _, _, _, _, _ = _reader()
    identity = _identity(binding)
    status = ObservationRecordV1(
        OBSERVATION_SCHEMA_VERSION,
        ObservationStreamRef(ObservationFamily.TRAINING,
                             identity["run"].project_ref, identity["run"].run_id),
        0, "2026-10-07T10:15:30Z",
        ObservationKind.TRAINING_PHASE_OBSERVED,
        TrainingPhaseObservedPayloadV1("metric_capture_exhausted"),
    )
    with pytest.raises(ValueError, match="sequence"):
        make_progress_file(
            **identity, index=0, previous_digest=ZERO_DIGEST, record=status,
        )
    content = {
        "schema_version": "synaptic-modal-packaged-progress/v1",
        "effect_id": identity["effect_id"],
        "command_digest": identity["command_digest"],
        "provider_job_ref": identity["provider_job_ref"],
        "execution_binding_digest": identity["execution_binding_digest"],
        "project_ref": identity["run"].project_ref,
        "run_id": identity["run"].run_id,
        "max_records": identity["max_records"],
        "index": 0,
        "previous_digest": ZERO_DIGEST,
        "kind": "exhausted",
        "entry": encode_training_progress_log(status, expected_run=identity["run"]).to_dict(),
    }
    raw = canonical_bytes({
        "content": content,
        "tag_hex": Auth().sign(PROGRESS_PURPOSE, canonical_bytes(content), "evidence-key").hex(),
    })
    with pytest.raises(ValueError, match="entry kind or sequence"):
        parse_progress_file(
            raw, verifier=Auth(), key_ref="evidence-key",
            effect_id=identity["effect_id"],
            command_digest=identity["command_digest"],
            provider_job_ref="fc-1",
            execution_binding_digest=identity["execution_binding_digest"],
            run=identity["run"], max_records=None,
            index=0, previous_digest=ZERO_DIGEST,
        )


def test_empty_page_rejects_forged_watermark_and_marker():
    binding, reader, _, _, _, _ = _reader()
    run = _identity(binding)["run"]
    page = reader.read_progress(
        binding, provider_job_ref="fc-1", expected_run=run,
        after_sequence=None, limit=2, maximum_bytes=4096,
    )
    assert page.entries == () and page.high_watermark_sequence == -1
    with pytest.raises(ValueError, match="empty page watermark"):
        replace(page, high_watermark_sequence=5, canonical_evidence=b"")
    with pytest.raises(ValueError, match="page marker"):
        replace(page, marker_digest="f" * 64, canonical_evidence=b"")
    with pytest.raises(ValueError, match="cursor anchor missing"):
        reader.read_progress(
            binding, provider_job_ref="fc-1", expected_run=run,
            after_sequence=10_000, limit=2, maximum_bytes=4096,
        )


def test_max_capture_status_is_pageable_at_sequence_10000():
    binding, reader, _, control, _, _ = _reader()
    reader._verifier = Auth()
    identity = _identity(binding)
    identity["max_records"] = 10_000
    stream = ObservationStreamRef(
        ObservationFamily.TRAINING,
        identity["run"].project_ref, identity["run"].run_id,
    )
    previous = "a" * 64
    for index in range(9984, 10_001):
        status = index == 10_000
        record = ObservationRecordV1(
            OBSERVATION_SCHEMA_VERSION, stream, index,
            "2026-10-07T10:15:30Z",
            ObservationKind.TRAINING_PHASE_OBSERVED if status
            else ObservationKind.TRAINING_METRIC_OBSERVED,
            TrainingPhaseObservedPayloadV1("metric_capture_exhausted")
            if status else metric(1),
        )
        raw, previous = make_progress_file(
            **identity, index=index, previous_digest=previous, record=record,
        )
        control.files[progress_path(identity["effect_id"], index)] = raw
    final_page = reader.read_progress(
        binding, provider_job_ref="fc-1", expected_run=identity["run"],
        after_sequence=9999, limit=2, maximum_bytes=4096,
    )
    assert [entry.sequence for entry in final_page.entries] == [10_000]
    assert final_page.exhausted is True
    empty_page = reader.read_progress(
        binding, provider_job_ref="fc-1", expected_run=identity["run"],
        after_sequence=10_000, limit=2, maximum_bytes=4096,
    )
    assert empty_page.entries == () and empty_page.exhausted is True
    assert empty_page.high_watermark_sequence == 10_000


@pytest.mark.skipif(sys.platform != "linux", reason="mounted_io requires Linux descriptor semantics")
def test_background_committed_prefix_pages_while_completion_is_absent(volume_layout, request):
    binding, reader, _, control, _, _ = _reader()
    identity = _identity(binding)
    reader._verifier = Auth()
    effect = identity["effect_id"]
    root = volume_layout[2]
    bound = root_tests._bind(volume_layout)
    request.addfinalizer(bound.close)
    publisher = PackagedProgressPublisher(
        **identity, claim_directory=bound.claim_directory,
        publish_exclusive=lambda stage, path, raw: bound.publish_exclusive(stage, path, raw, 8192),
    )
    for step in range(1, 4):
        publisher.on_metric_frame(metric(step))
    publisher.on_metric_exhausted()
    publisher.close()
    publisher._thread.join(timeout=3)
    assert not publisher._thread.is_alive()
    assert not publisher._failed.is_set()

    # The reader sees no local mounted writes until a provider background
    # commit exposes a stable snapshot through its SDK Volume read methods.
    completion = f"operations/{effect}/evidence/packaged-completion.json"
    control.files.pop(completion)
    control.files.pop(completion.removesuffix(".json") + ".mac")
    absent = reader.read_progress(
        binding, provider_job_ref="fc-1", expected_run=identity["run"],
        after_sequence=None, limit=2, maximum_bytes=4096,
    )
    assert absent.entries == () and absent.high_watermark_sequence == -1

    first_path = progress_path(effect, 0)
    control.files[first_path] = mounted_io.read_regular(root, root / first_path, 8192)
    partial = reader.read_progress(
        binding, provider_job_ref="fc-1", expected_run=identity["run"],
        after_sequence=None, limit=2, maximum_bytes=4096,
    )
    assert [entry.sequence for entry in partial.entries] == [0]
    assert partial.truncated is False and partial.exhausted is False

    # A background snapshot of a still-writing final leaf is not treated as
    # an authenticated absent tail. It closes the read instead of exporting it.
    torn_path = progress_path(effect, 1)
    control.files[torn_path] = mounted_io.read_regular(root, root / torn_path, 8192)[:32]
    with pytest.raises(ValueError):
        reader.read_progress(
            binding, provider_job_ref="fc-1", expected_run=identity["run"],
            after_sequence=None, limit=2, maximum_bytes=4096,
        )
    control.files.pop(torn_path)

    for index in range(1, 4):
        path = progress_path(effect, index)
        control.files[path] = mounted_io.read_regular(root, root / path, 8192)
    first = reader.read_progress(
        binding, provider_job_ref="fc-1", expected_run=identity["run"],
        after_sequence=None, limit=2, maximum_bytes=4096,
    )
    assert [entry.sequence for entry in first.entries] == [0, 1]
    assert first.truncated is True and first.exhausted is False
    assert first.high_watermark_sequence == 1 and first.marker_digest is not None
    assert first.canonical_evidence == first.expected_evidence
    second = reader.read_progress(
        binding, provider_job_ref="fc-1", expected_run=identity["run"],
        after_sequence=1, limit=2, maximum_bytes=4096,
    )
    assert [entry.sequence for entry in second.entries] == [2, 3]
    assert second.truncated is False and second.exhausted is True
    assert second.high_watermark_sequence == 3
    assert second.marker_digest != first.marker_digest
    status = decode_training_progress_log(second.entries[-1], expected_run=identity["run"])
    assert status.payload.phase == "metric_capture_exhausted"
    assert [entry.sequence for entry in (*first.entries, *second.entries)] == [0, 1, 2, 3]
    after_status = reader.read_progress(
        binding, provider_job_ref="fc-1", expected_run=identity["run"],
        after_sequence=3, limit=2, maximum_bytes=4096,
    )
    assert after_status.entries == () and after_status.exhausted is True
    assert after_status.high_watermark_sequence == 3

    # A signed cursor cannot silently skip an absent anchor; forged content
    # also fails authentication without exposing the rejected metric bytes.
    anchor = progress_path(effect, 1)
    saved = control.files.pop(anchor)
    with pytest.raises(ValueError, match="inventory gap|cursor anchor missing"):
        reader.read_progress(
            binding, provider_job_ref="fc-1", expected_run=identity["run"],
            after_sequence=1, limit=2, maximum_bytes=4096,
        )
    control.files[anchor] = saved.replace(b"0.00015", b"0.00016")
    with pytest.raises(ValueError, match="authentication"):
        reader.read_progress(
            binding, provider_job_ref="fc-1", expected_run=identity["run"],
            after_sequence=1, limit=2, maximum_bytes=4096,
        )


@pytest.mark.skipif(sys.platform != "linux", reason="mounted_io requires Linux descriptor semantics")
def test_progress_write_failure_does_not_raise_into_metric_callback(volume_layout, request):
    binding, _, _, _, _, _ = _reader()
    identity = _identity(binding)
    root = volume_layout[2]
    bound = root_tests._bind(volume_layout)
    request.addfinalizer(bound.close)
    calls = []

    def fail_second(stage, path, raw):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("private Volume failure")
        bound.publish_exclusive(stage, path, raw, 8192)

    publisher = PackagedProgressPublisher(
        **identity, claim_directory=bound.claim_directory,
        publish_exclusive=fail_second,
    )
    publisher.on_metric_frame(metric(1))
    publisher.on_metric_frame(metric(2))
    publisher.close()
    publisher._thread.join(timeout=3)
    assert publisher._failed.is_set()
    assert len(calls) == 2
    assert mounted_io.read_regular(root, root / calls[0], 8192)


def test_queue_overflow_does_not_claim_capture_budget_exhaustion():
    binding, _, _, _, _, _ = _reader()
    identity = _identity(binding)
    identity["max_records"] = 1000
    writing, release = Event(), Event()
    written = []

    def slow_write(stage, path, raw):
        writing.set()
        assert release.wait(5)
        written.append(path)

    publisher = PackagedProgressPublisher(
        **identity, claim_directory=lambda path: None, publish_exclusive=slow_write,
    )
    try:
        publisher.on_metric_frame(metric(1))
        assert writing.wait(5)
        for _ in range(257):
            publisher.on_metric_frame(metric(1))
        assert publisher._failed.is_set()
        publisher.on_metric_exhausted()
    finally:
        release.set()
        publisher.close()
        publisher._thread.join(timeout=10)
    assert not publisher._thread.is_alive()
    assert written == [progress_path(identity["effect_id"], i) for i in range(257)]
