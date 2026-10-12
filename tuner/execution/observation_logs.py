"""Bounded training observations projected onto the existing run-log wire shape.

This internal codec does not authenticate a producer. Callers must obtain the
expected run and log entries through the authenticated execution/read boundary.
It neither enables provider logs nor assigns, renumbers or persists sequences.
Progress additionally admits one fixed capture-exhausted phase; arbitrary phase
text is never exported. The metric-only codec preserves its original contract.
"""

from __future__ import annotations

import json

from synaptic_tuner.api.v1._contract import canonical_bytes
from synaptic_tuner.api.v1.observations import (
    ObservationFamily,
    ObservationKind,
    ObservationRecordV1,
)
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.runs_facade import RunLogEntry, RunLogLevel


_MAX_BYTES = 4096
_ERROR = "invalid training metric log projection"


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(_ERROR)
        result[key] = value
    return result


def _reject_constant(_value: str) -> object:
    raise ValueError(_ERROR)


def _bound_record(record: ObservationRecordV1, run: TrainingRunRef, *,
                  allow_capture_exhausted: bool = False) -> ObservationRecordV1:
    if type(run) is not TrainingRunRef or type(record) is not ObservationRecordV1:
        raise ValueError(_ERROR)
    run = TrainingRunRef.from_dict(run.to_dict())
    record = ObservationRecordV1.from_dict(record.to_dict())
    admitted_kind = record.kind is ObservationKind.TRAINING_METRIC_OBSERVED or (
        allow_capture_exhausted
        and record.kind is ObservationKind.TRAINING_PHASE_OBSERVED
        and record.payload.to_dict() == {"phase": "metric_capture_exhausted"}
    )
    if (
        not admitted_kind
        or record.stream.family is not ObservationFamily.TRAINING
        or record.stream.project_ref != run.project_ref
        or record.stream.entity_id != run.run_id
        or not record.occurred_at.endswith("Z")
    ):
        raise ValueError(_ERROR)
    # Bound inherited strings before serialization. This projection does not
    # change the broader observation contract's legacy identifier/time policy.
    if any(len(value) > _MAX_BYTES for value in (
        record.stream.project_ref, record.stream.entity_id, record.occurred_at,
    )):
        raise ValueError(_ERROR)
    return record


def encode_training_metric_log(
    record: ObservationRecordV1, *, expected_run: TrainingRunRef,
) -> RunLogEntry:
    """Encode a numeric observation, preserving its exact sequence and UTC time.

    The full canonical record, including stream identity, lives in the message.
    The inherited outer fields must agree, not substitute another identity.
    Legacy observation offsets remain legal elsewhere; this chart projection
    requires UTC ``Z`` without silently converting a timestamp.
    """
    return _encode_training_log(record, expected_run=expected_run)


def _encode_training_log(record, *, expected_run, allow_capture_exhausted=False):
    try:
        record = _bound_record(record, expected_run,
                               allow_capture_exhausted=allow_capture_exhausted)
        raw = canonical_bytes(record.to_dict())
        if len(raw) > _MAX_BYTES:
            raise ValueError(_ERROR)
        return RunLogEntry(
            sequence=record.sequence,
            timestamp=record.occurred_at,
            level=RunLogLevel.INFO,
            event=record.kind.value,
            message=raw.decode("utf-8"),
            size_bytes=len(raw),
        )
    except (ValueError, TypeError, AttributeError, KeyError, OverflowError, RecursionError):
        raise ValueError(_ERROR) from None


def decode_training_metric_log(
    entry: RunLogEntry, *, expected_run: TrainingRunRef,
) -> ObservationRecordV1:
    """Decode an already authenticated log entry with strict canonical checks.

    Wrong stream, kind, sequence, timestamp, level or byte count is rejected.
    Invalid input never appears in the closed diagnostic. Canonical equality
    rejects alternate JSON encodings as well as unknown/missing fields.
    """
    return _decode_training_log(entry, expected_run=expected_run)


def _decode_training_log(entry, *, expected_run, allow_capture_exhausted=False):
    try:
        if type(entry) is not RunLogEntry:
            raise ValueError(_ERROR)
        # Reject oversized text before parsing or reconstructing a forged entry.
        if type(entry.message) is not str or len(entry.message) > _MAX_BYTES:
            raise ValueError(_ERROR)
        entry = RunLogEntry.from_dict(entry.to_dict())
        raw = entry.message.encode("utf-8")
        document = json.loads(
            raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant,
        )
        record = _bound_record(ObservationRecordV1.from_dict(document), expected_run,
                               allow_capture_exhausted=allow_capture_exhausted)
        rebuilt = _encode_training_log(record, expected_run=expected_run,
                                       allow_capture_exhausted=allow_capture_exhausted)
        if rebuilt != entry:
            raise ValueError(_ERROR)
        return record
    except (ValueError, TypeError, AttributeError, KeyError, OverflowError, RecursionError):
        raise ValueError(_ERROR) from None


def encode_training_progress_log(
    record: ObservationRecordV1, *, expected_run: TrainingRunRef,
) -> RunLogEntry:
    """Admit metrics or the one closed, parent-issued capture budget status."""
    return _encode_training_log(record, expected_run=expected_run,
                                allow_capture_exhausted=True)


def decode_training_progress_log(
    entry: RunLogEntry, *, expected_run: TrainingRunRef,
) -> ObservationRecordV1:
    """Read numeric progress or exactly metric_capture_exhausted, never prose."""
    return _decode_training_log(entry, expected_run=expected_run,
                                allow_capture_exhausted=True)


__all__: list[str] = []
