"""Pure projection of one public training metric into a generic run log."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from synaptic_tuner.api.v1._contract import canonical_bytes
from synaptic_tuner.api.v1.observations import (
    OBSERVATION_SCHEMA_VERSION,
    ObservationFamily,
    ObservationKind,
    ObservationRecordV1,
    ObservationStreamRef,
    TrainingMetricObservedPayloadV1,
    TrainingPhaseObservedPayloadV1,
)
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.runs_facade import RunLogLevel
from tuner.execution.observation_logs import (
    decode_training_metric_log,
    encode_training_metric_log,
    decode_training_progress_log,
    encode_training_progress_log,
)


RUN = TrainingRunRef("run-metric-1", "project-metric-1")
AT = "2026-10-07T10:15:30Z"
ERROR = "invalid training metric log projection"


def metric_record(
    sequence: int = 0, *, run: TrainingRunRef = RUN,
    occurred_at: str = AT,
) -> ObservationRecordV1:
    return ObservationRecordV1(
        OBSERVATION_SCHEMA_VERSION,
        ObservationStreamRef(ObservationFamily.TRAINING, run.project_ref, run.run_id),
        sequence,
        occurred_at,
        ObservationKind.TRAINING_METRIC_OBSERVED,
        TrainingMetricObservedPayloadV1(
            step=12,
            total_steps=25,
            loss=0.125,
            learning_rate=0.00015,
            epoch=0.48,
            train_elapsed_seconds=30.25,
            job_elapsed_seconds=42.5,
            gpu_current_allocated_bytes=1024,
            gpu_current_reserved_bytes=2048,
            gpu_peak_allocated_bytes=4096,
            gpu_peak_reserved_bytes=8192,
            gpu_utilization_percent=72.5,
            tokens_per_second=103.25,
            estimated_samples_per_second=2.75,
        ),
    )


def test_training_metric_projection_round_trips_complete_canonical_record() -> None:
    record = metric_record(sequence=7)
    entry = encode_training_metric_log(record, expected_run=RUN)

    assert entry.sequence == record.sequence
    assert entry.timestamp == record.occurred_at
    assert entry.level is RunLogLevel.INFO
    assert entry.event == ObservationKind.TRAINING_METRIC_OBSERVED.value
    assert entry.message.encode("utf-8") == canonical_bytes(record.to_dict())
    assert entry.size_bytes == len(entry.message.encode("utf-8"))
    assert decode_training_metric_log(entry, expected_run=RUN) == record
    assert decode_training_metric_log(entry, expected_run=RUN).payload.learning_rate == 0.00015


def test_progress_projects_only_closed_capture_status_without_widening_metric_codec():
    record = replace(metric_record(), kind=ObservationKind.TRAINING_PHASE_OBSERVED,
                     payload=TrainingPhaseObservedPayloadV1("metric_capture_exhausted"))
    entry = encode_training_progress_log(record, expected_run=RUN)
    assert decode_training_progress_log(entry, expected_run=RUN) == record
    with pytest.raises(ValueError):
        encode_training_metric_log(record, expected_run=RUN)
    with pytest.raises(ValueError):
        decode_training_metric_log(entry, expected_run=RUN)
    for phase in ("PRIVATE_SENTINEL", "training_completed", "metric_capture_exhausted "):
        with pytest.raises(ValueError):
            encode_training_progress_log(replace(record,
                payload=TrainingPhaseObservedPayloadV1(phase)), expected_run=RUN)


def test_projection_accepts_utc_z_and_exact_4096_byte_message() -> None:
    initial = metric_record(run=TrainingRunRef("r", "p"))
    remaining = 4096 - len(canonical_bytes(initial.to_dict()))
    run = TrainingRunRef("r", "p" + "x" * remaining)
    record = metric_record(run=run)
    entry = encode_training_metric_log(record, expected_run=run)

    assert entry.timestamp.endswith("Z")
    assert entry.size_bytes == 4096
    assert decode_training_metric_log(entry, expected_run=run) == record


def test_projection_rejects_over_4096_utf8_bytes_on_encode() -> None:
    initial = metric_record(run=TrainingRunRef("r", "p"))
    remaining = 4096 - len(canonical_bytes(initial.to_dict()))
    run = TrainingRunRef("r", "p" + "x" * (remaining + 1))

    with pytest.raises(ValueError, match=f"^{ERROR}$"):
        encode_training_metric_log(metric_record(run=run), expected_run=run)


def test_projection_requires_utc_z_at_its_boundary() -> None:
    record = metric_record(occurred_at="2026-10-07T06:15:30-04:00")
    with pytest.raises(ValueError, match=f"^{ERROR}$"):
        encode_training_metric_log(record, expected_run=RUN)


@pytest.mark.parametrize("change", [
    {"kind": ObservationKind.TRAINING_PHASE_OBSERVED,
     "payload": TrainingPhaseObservedPayloadV1("training")},
    {"stream": ObservationStreamRef(ObservationFamily.TRAINING, "other-project", RUN.run_id)},
    {"stream": ObservationStreamRef(ObservationFamily.TRAINING, RUN.project_ref, "other-run")},
])
def test_encode_rejects_wrong_kind_or_run(change: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=f"^{ERROR}$"):
        encode_training_metric_log(replace(metric_record(), **change), expected_run=RUN)


@pytest.mark.parametrize("change", [
    {"sequence": 1},
    {"timestamp": "2026-10-07T10:15:31Z"},
    {"level": RunLogLevel.WARNING},
    {"event": "training_progress"},
    {"message": "{}"},
])
def test_decode_rejects_outer_entry_disagreement(change: dict[str, object]) -> None:
    entry = encode_training_metric_log(metric_record(), expected_run=RUN)
    if "message" in change:
        change["size_bytes"] = len(change["message"].encode("utf-8"))
    with pytest.raises(ValueError, match=f"^{ERROR}$"):
        decode_training_metric_log(replace(entry, **change), expected_run=RUN)


@pytest.mark.parametrize("change", [
    {"stream": {"family": "training", "project_ref": "other", "entity_id": RUN.run_id}},
    {"stream": {"family": "training", "project_ref": RUN.project_ref, "entity_id": "other"}},
    {"kind": "training_phase_observed", "payload": {"phase": "training"}},
    {"sequence": 1},
    {"occurred_at": "2026-10-07T10:15:31Z"},
])
def test_decode_rejects_inner_record_disagreement(change: dict[str, object]) -> None:
    entry = encode_training_metric_log(metric_record(), expected_run=RUN)
    document = json.loads(entry.message)
    document.update(change)
    message = canonical_bytes(document).decode("utf-8")

    with pytest.raises(ValueError, match=f"^{ERROR}$"):
        decode_training_metric_log(
            replace(entry, message=message, size_bytes=len(message.encode("utf-8"))),
            expected_run=RUN,
        )


@pytest.mark.parametrize("mutate", [
    lambda message: message.replace('"sequence":0', '"sequence":0,"sequence":0', 1),
    lambda message: message.replace('"loss":0.125', '"loss":NaN', 1),
    lambda message: message.replace('"loss":0.125', '"loss":Infinity', 1),
    lambda message: message.replace('"step":12', '"step":' + '[' * 1500 + '12' + ']' * 1500, 1),
])
def test_decode_rejects_duplicate_nonfinite_or_excessively_nested_json(mutate) -> None:
    entry = encode_training_metric_log(metric_record(), expected_run=RUN)
    message = mutate(entry.message)
    assert len(message.encode("utf-8")) <= 4096
    with pytest.raises(ValueError, match=f"^{ERROR}$"):
        decode_training_metric_log(
            replace(entry, message=message, size_bytes=len(message.encode("utf-8"))),
            expected_run=RUN,
        )


def test_decode_rejects_noncanonical_json_and_invalid_utf8_scalar() -> None:
    entry = encode_training_metric_log(metric_record(), expected_run=RUN)
    noncanonical = json.dumps(json.loads(entry.message), ensure_ascii=True)
    assert noncanonical != entry.message
    with pytest.raises(ValueError, match=f"^{ERROR}$"):
        decode_training_metric_log(
            replace(entry, message=noncanonical, size_bytes=len(noncanonical.encode("utf-8"))),
            expected_run=RUN,
        )

    object.__setattr__(entry, "message", "\ud800")
    with pytest.raises(ValueError, match=f"^{ERROR}$"):
        decode_training_metric_log(entry, expected_run=RUN)


def test_decode_rejects_tampered_oversize_and_noncanonical_inner_field() -> None:
    entry = encode_training_metric_log(metric_record(), expected_run=RUN)
    object.__setattr__(entry, "message", "x" * 4097)
    object.__setattr__(entry, "size_bytes", 4097)
    with pytest.raises(ValueError, match=f"^{ERROR}$"):
        decode_training_metric_log(entry, expected_run=RUN)

    entry = encode_training_metric_log(metric_record(), expected_run=RUN)
    document = json.loads(entry.message)
    document["payload"]["learning_rate"] = "0.00015"
    message = canonical_bytes(document).decode("utf-8")
    with pytest.raises(ValueError, match=f"^{ERROR}$"):
        decode_training_metric_log(
            replace(entry, message=message, size_bytes=len(message.encode("utf-8"))),
            expected_run=RUN,
        )


def test_decode_rejects_tampered_dataclass_and_wrong_expected_run() -> None:
    entry = encode_training_metric_log(metric_record(), expected_run=RUN)
    with pytest.raises(ValueError, match=f"^{ERROR}$"):
        decode_training_metric_log(entry, expected_run=TrainingRunRef("other", RUN.project_ref))
    object.__setattr__(entry, "size_bytes", entry.size_bytes + 1)
    with pytest.raises(ValueError, match=f"^{ERROR}$"):
        decode_training_metric_log(entry, expected_run=RUN)


def test_encode_rejects_tampered_observation_dataclass() -> None:
    record = metric_record()
    object.__setattr__(record.payload, "learning_rate", float("nan"))
    with pytest.raises(ValueError, match=f"^{ERROR}$"):
        encode_training_metric_log(record, expected_run=RUN)
