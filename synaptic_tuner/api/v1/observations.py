"""Typed, append-only observation stream contracts shared by every facade.

Location: ``synaptic_tuner/api/v1/observations.py``.

Progress never enters the lifecycle record. Each family emits observations
into a stream keyed by ``(family, project_ref, entity_id)``; the ``kind`` of a
record is closed per family and selects one exact payload type. Observations
carry no authority: none can authorize an effect, change a run state or
satisfy a verification, so a host may truncate or discard a stream freely.

The read path reuses the paging discipline of ``runs_facade.py``: an
``ObservationsRequest`` names a stream, an optional ``after_sequence`` and a
bounded ``limit``; an ``ObservationPage`` binds its request, keeps sequences
strictly increasing, and keeps ``next_cursor`` and ``truncated`` in agreement.

Contract only: this module imports nothing from ``tuner.*`` and is registered
in both import-closure gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ._contract import digest_text, exact_fields, exact_integer, required_text
from ._timestamps import require_rfc3339


OBSERVATION_SCHEMA_VERSION = "synaptic-observation/v1"

_MAX_SEQUENCE = 2**63 - 1
_MAX_LIMIT = 200
_MAX_TOKEN_TEXT_BYTES = 4096


def _text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    return required_text(value, name)


def _bounded_integer(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} through {maximum}")
    return value


def _bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact boolean")
    return value


def _finite_float(value: object, name: str) -> float:
    if type(value) is bool or type(value) not in (int, float):
        raise TypeError(f"{name} must be a number")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return number


def _closed_text(value: object, name: str, allowed: frozenset[str]) -> str:
    value = _text(value, name)
    if value not in allowed:
        raise ValueError(f"{name} must be one of: {', '.join(sorted(allowed))}")
    return value


class ObservationFamily(str, Enum):
    TRAINING = "training"
    EVALUATION = "evaluation"
    CHAT = "chat"
    DATA = "data"
    PIPELINE = "pipeline"


class ObservationKind(str, Enum):
    """Closed per-family vocabulary; each member selects one payload type."""

    TRAINING_PHASE_OBSERVED = "training_phase_observed"
    EVALUATION_CASE_STARTED = "evaluation_case_started"
    EVALUATION_CASE_SCORED = "evaluation_case_scored"
    EVALUATION_STAGE_COMPLETED = "evaluation_stage_completed"
    CHAT_TURN_STARTED = "chat_turn_started"
    CHAT_TURN_COMPLETED = "chat_turn_completed"
    CHAT_TOKEN = "chat_token"
    DATA_ROW_WRITTEN = "data_row_written"
    DATA_STAGE_GATE_EVALUATED = "data_stage_gate_evaluated"
    DATA_SCENARIO_COMPLETED = "data_scenario_completed"
    PIPELINE_STAGE_STARTED = "pipeline_stage_started"
    PIPELINE_STAGE_COMPLETED = "pipeline_stage_completed"

    @property
    def family(self) -> ObservationFamily:
        return _KIND_FAMILIES[self]


@dataclass(frozen=True, slots=True)
class ObservationStreamRef:
    family: ObservationFamily
    project_ref: str
    entity_id: str

    def __post_init__(self) -> None:
        if type(self.family) is not ObservationFamily:
            raise TypeError("family must be exact ObservationFamily")
        object.__setattr__(self, "project_ref", _text(self.project_ref, "project_ref"))
        object.__setattr__(self, "entity_id", _text(self.entity_id, "entity_id"))

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family.value,
            "project_ref": self.project_ref,
            "entity_id": self.entity_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ObservationStreamRef":
        value = exact_fields(
            value, frozenset({"family", "project_ref", "entity_id"}), "observation_stream"
        )
        family = _text(value["family"], "family")
        try:
            parsed = ObservationFamily(family)
        except ValueError:
            raise ValueError("unknown observation family") from None
        return cls(parsed, value["project_ref"], value["entity_id"])  # type: ignore[arg-type]


# --- payloads -------------------------------------------------------------
#
# One frozen slotted dataclass per kind. Field sets are exact: ``from_dict``
# refuses unknown and missing fields, and ``to_dict`` emits every field.

EVALUATION_VERDICTS = frozenset({"passed", "failed", "inconclusive"})
PIPELINE_STAGE_OUTCOMES = frozenset({"succeeded", "failed", "skipped", "cancelled"})


@dataclass(frozen=True, slots=True)
class TrainingPhaseObservedPayloadV1:
    phase: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", _text(self.phase, "phase"))

    def to_dict(self) -> dict[str, object]:
        return {"phase": self.phase}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "TrainingPhaseObservedPayloadV1":
        value = exact_fields(value, frozenset({"phase"}), "training_phase_observed")
        return cls(value["phase"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class EvaluationCaseStartedPayloadV1:
    case_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_ref", _text(self.case_ref, "case_ref"))

    def to_dict(self) -> dict[str, object]:
        return {"case_ref": self.case_ref}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "EvaluationCaseStartedPayloadV1":
        value = exact_fields(value, frozenset({"case_ref"}), "evaluation_case_started")
        return cls(value["case_ref"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class EvaluationCaseScoredPayloadV1:
    case_ref: str
    verdict: str
    score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_ref", _text(self.case_ref, "case_ref"))
        object.__setattr__(self, "verdict", _closed_text(self.verdict, "verdict", EVALUATION_VERDICTS))
        object.__setattr__(self, "score", _finite_float(self.score, "score"))

    def to_dict(self) -> dict[str, object]:
        return {"case_ref": self.case_ref, "verdict": self.verdict, "score": self.score}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "EvaluationCaseScoredPayloadV1":
        value = exact_fields(
            value, frozenset({"case_ref", "verdict", "score"}), "evaluation_case_scored"
        )
        return cls(value["case_ref"], value["verdict"], value["score"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class EvaluationStageCompletedPayloadV1:
    stage: str
    cases_scored: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _text(self.stage, "stage"))
        object.__setattr__(self, "cases_scored", exact_integer(self.cases_scored, "cases_scored"))

    def to_dict(self) -> dict[str, object]:
        return {"stage": self.stage, "cases_scored": self.cases_scored}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "EvaluationStageCompletedPayloadV1":
        value = exact_fields(
            value, frozenset({"stage", "cases_scored"}), "evaluation_stage_completed"
        )
        return cls(value["stage"], value["cases_scored"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ChatTurnStartedPayloadV1:
    request_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", exact_integer(self.request_id, "request_id"))

    def to_dict(self) -> dict[str, object]:
        return {"request_id": self.request_id}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ChatTurnStartedPayloadV1":
        value = exact_fields(value, frozenset({"request_id"}), "chat_turn_started")
        return cls(value["request_id"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ChatTurnCompletedPayloadV1:
    request_id: int
    content_digest: str
    content_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", exact_integer(self.request_id, "request_id"))
        object.__setattr__(self, "content_digest", digest_text(self.content_digest, "content_digest"))
        object.__setattr__(self, "content_bytes", exact_integer(self.content_bytes, "content_bytes"))

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "content_digest": self.content_digest,
            "content_bytes": self.content_bytes,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ChatTurnCompletedPayloadV1":
        value = exact_fields(
            value,
            frozenset({"request_id", "content_digest", "content_bytes"}),
            "chat_turn_completed",
        )
        return cls(value["request_id"], value["content_digest"], value["content_bytes"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ChatTokenPayloadV1:
    """Reserved for token streaming against the same turn identity; not emitted yet."""

    request_id: int
    token_index: int
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", exact_integer(self.request_id, "request_id"))
        object.__setattr__(self, "token_index", exact_integer(self.token_index, "token_index"))
        if type(self.text) is not str:
            raise TypeError("text must be an exact string")
        if len(self.text.encode("utf-8")) > _MAX_TOKEN_TEXT_BYTES:
            raise ValueError(f"text exceeds {_MAX_TOKEN_TEXT_BYTES} UTF-8 bytes")

    def to_dict(self) -> dict[str, object]:
        return {"request_id": self.request_id, "token_index": self.token_index, "text": self.text}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ChatTokenPayloadV1":
        value = exact_fields(value, frozenset({"request_id", "token_index", "text"}), "chat_token")
        return cls(value["request_id"], value["token_index"], value["text"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class DataRowWrittenPayloadV1:
    scenario_ref: str
    rows_written: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_ref", _text(self.scenario_ref, "scenario_ref"))
        object.__setattr__(self, "rows_written", exact_integer(self.rows_written, "rows_written", minimum=1))

    def to_dict(self) -> dict[str, object]:
        return {"scenario_ref": self.scenario_ref, "rows_written": self.rows_written}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "DataRowWrittenPayloadV1":
        value = exact_fields(value, frozenset({"scenario_ref", "rows_written"}), "data_row_written")
        return cls(value["scenario_ref"], value["rows_written"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class DataStageGateEvaluatedPayloadV1:
    scenario_ref: str
    gate: str
    passed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_ref", _text(self.scenario_ref, "scenario_ref"))
        object.__setattr__(self, "gate", _text(self.gate, "gate"))
        object.__setattr__(self, "passed", _bool(self.passed, "passed"))

    def to_dict(self) -> dict[str, object]:
        return {"scenario_ref": self.scenario_ref, "gate": self.gate, "passed": self.passed}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "DataStageGateEvaluatedPayloadV1":
        value = exact_fields(
            value, frozenset({"scenario_ref", "gate", "passed"}), "data_stage_gate_evaluated"
        )
        return cls(value["scenario_ref"], value["gate"], value["passed"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class DataScenarioCompletedPayloadV1:
    scenario_ref: str
    rows_written: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_ref", _text(self.scenario_ref, "scenario_ref"))
        object.__setattr__(self, "rows_written", exact_integer(self.rows_written, "rows_written"))

    def to_dict(self) -> dict[str, object]:
        return {"scenario_ref": self.scenario_ref, "rows_written": self.rows_written}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "DataScenarioCompletedPayloadV1":
        value = exact_fields(
            value, frozenset({"scenario_ref", "rows_written"}), "data_scenario_completed"
        )
        return cls(value["scenario_ref"], value["rows_written"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class PipelineStageStartedPayloadV1:
    stage: str
    attempt_key: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _text(self.stage, "stage"))
        object.__setattr__(self, "attempt_key", digest_text(self.attempt_key, "attempt_key"))

    def to_dict(self) -> dict[str, object]:
        return {"stage": self.stage, "attempt_key": self.attempt_key}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PipelineStageStartedPayloadV1":
        value = exact_fields(value, frozenset({"stage", "attempt_key"}), "pipeline_stage_started")
        return cls(value["stage"], value["attempt_key"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class PipelineStageCompletedPayloadV1:
    stage: str
    attempt_key: str
    outcome: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _text(self.stage, "stage"))
        object.__setattr__(self, "attempt_key", digest_text(self.attempt_key, "attempt_key"))
        object.__setattr__(self, "outcome", _closed_text(self.outcome, "outcome", PIPELINE_STAGE_OUTCOMES))

    def to_dict(self) -> dict[str, object]:
        return {"stage": self.stage, "attempt_key": self.attempt_key, "outcome": self.outcome}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PipelineStageCompletedPayloadV1":
        value = exact_fields(
            value, frozenset({"stage", "attempt_key", "outcome"}), "pipeline_stage_completed"
        )
        return cls(value["stage"], value["attempt_key"], value["outcome"])  # type: ignore[arg-type]


_KIND_FAMILIES: dict[ObservationKind, ObservationFamily] = {
    ObservationKind.TRAINING_PHASE_OBSERVED: ObservationFamily.TRAINING,
    ObservationKind.EVALUATION_CASE_STARTED: ObservationFamily.EVALUATION,
    ObservationKind.EVALUATION_CASE_SCORED: ObservationFamily.EVALUATION,
    ObservationKind.EVALUATION_STAGE_COMPLETED: ObservationFamily.EVALUATION,
    ObservationKind.CHAT_TURN_STARTED: ObservationFamily.CHAT,
    ObservationKind.CHAT_TURN_COMPLETED: ObservationFamily.CHAT,
    ObservationKind.CHAT_TOKEN: ObservationFamily.CHAT,
    ObservationKind.DATA_ROW_WRITTEN: ObservationFamily.DATA,
    ObservationKind.DATA_STAGE_GATE_EVALUATED: ObservationFamily.DATA,
    ObservationKind.DATA_SCENARIO_COMPLETED: ObservationFamily.DATA,
    ObservationKind.PIPELINE_STAGE_STARTED: ObservationFamily.PIPELINE,
    ObservationKind.PIPELINE_STAGE_COMPLETED: ObservationFamily.PIPELINE,
}

OBSERVATION_PAYLOAD_TYPES: dict[ObservationKind, type] = {
    ObservationKind.TRAINING_PHASE_OBSERVED: TrainingPhaseObservedPayloadV1,
    ObservationKind.EVALUATION_CASE_STARTED: EvaluationCaseStartedPayloadV1,
    ObservationKind.EVALUATION_CASE_SCORED: EvaluationCaseScoredPayloadV1,
    ObservationKind.EVALUATION_STAGE_COMPLETED: EvaluationStageCompletedPayloadV1,
    ObservationKind.CHAT_TURN_STARTED: ChatTurnStartedPayloadV1,
    ObservationKind.CHAT_TURN_COMPLETED: ChatTurnCompletedPayloadV1,
    ObservationKind.CHAT_TOKEN: ChatTokenPayloadV1,
    ObservationKind.DATA_ROW_WRITTEN: DataRowWrittenPayloadV1,
    ObservationKind.DATA_STAGE_GATE_EVALUATED: DataStageGateEvaluatedPayloadV1,
    ObservationKind.DATA_SCENARIO_COMPLETED: DataScenarioCompletedPayloadV1,
    ObservationKind.PIPELINE_STAGE_STARTED: PipelineStageStartedPayloadV1,
    ObservationKind.PIPELINE_STAGE_COMPLETED: PipelineStageCompletedPayloadV1,
}

if set(_KIND_FAMILIES) != set(ObservationKind) or set(OBSERVATION_PAYLOAD_TYPES) != set(ObservationKind):  # pragma: no cover - module invariant
    raise RuntimeError("every observation kind must bind exactly one family and one payload type")


def _parse_kind(value: object) -> ObservationKind:
    value = _text(value, "kind")
    try:
        return ObservationKind(value)
    except ValueError:
        raise ValueError("unknown observation kind") from None


@dataclass(frozen=True, slots=True)
class ObservationRecordV1:
    schema_version: str
    stream: ObservationStreamRef
    sequence: int
    occurred_at: str
    kind: ObservationKind
    payload: object

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != OBSERVATION_SCHEMA_VERSION:
            raise ValueError("unsupported observation schema version")
        if type(self.stream) is not ObservationStreamRef:
            raise TypeError("stream must be exact ObservationStreamRef")
        object.__setattr__(self, "stream", ObservationStreamRef.from_dict(self.stream.to_dict()))
        _bounded_integer(self.sequence, "sequence", minimum=0, maximum=_MAX_SEQUENCE)
        object.__setattr__(self, "occurred_at", require_rfc3339(self.occurred_at, "occurred_at"))
        if type(self.kind) is not ObservationKind:
            raise TypeError("kind must be exact ObservationKind")
        if self.kind.family is not self.stream.family:
            raise ValueError("observation kind does not belong to the stream family")
        payload_type = OBSERVATION_PAYLOAD_TYPES[self.kind]
        if type(self.payload) is not payload_type:
            raise TypeError(f"payload must be exact {payload_type.__name__}")
        object.__setattr__(self, "payload", payload_type.from_dict(self.payload.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "stream": self.stream.to_dict(),
            "sequence": self.sequence,
            "occurred_at": self.occurred_at,
            "kind": self.kind.value,
            "payload": self.payload.to_dict(),  # type: ignore[attr-defined]
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ObservationRecordV1":
        value = exact_fields(
            value,
            frozenset({"schema_version", "stream", "sequence", "occurred_at", "kind", "payload"}),
            "observation_record",
        )
        kind = _parse_kind(value["kind"])
        payload = OBSERVATION_PAYLOAD_TYPES[kind].from_dict(value["payload"])  # type: ignore[arg-type]
        return cls(
            _text(value["schema_version"], "schema_version"),
            ObservationStreamRef.from_dict(value["stream"]),  # type: ignore[arg-type]
            value["sequence"],  # type: ignore[arg-type]
            _text(value["occurred_at"], "occurred_at"),
            kind,
            payload,
        )


@dataclass(frozen=True, slots=True)
class ObservationsRequest:
    stream: ObservationStreamRef
    after_sequence: int | None = None
    limit: int = 100

    def __post_init__(self) -> None:
        if type(self.stream) is not ObservationStreamRef:
            raise TypeError("stream must be exact ObservationStreamRef")
        object.__setattr__(self, "stream", ObservationStreamRef.from_dict(self.stream.to_dict()))
        if self.after_sequence is not None:
            _bounded_integer(self.after_sequence, "after_sequence", minimum=0, maximum=_MAX_SEQUENCE)
        _bounded_integer(self.limit, "limit", minimum=1, maximum=_MAX_LIMIT)


@dataclass(frozen=True, slots=True)
class ObservationPage:
    request: ObservationsRequest
    records: tuple[ObservationRecordV1, ...]
    next_cursor: int | None = None
    truncated: bool = False

    def __post_init__(self) -> None:
        if type(self.request) is not ObservationsRequest:
            raise TypeError("request must be exact ObservationsRequest")
        if type(self.records) is not tuple or any(type(item) is not ObservationRecordV1 for item in self.records):
            raise TypeError("records must be an exact tuple of ObservationRecordV1")
        if len(self.records) > self.request.limit:
            raise ValueError("records exceed requested limit")
        if any(item.stream != self.request.stream for item in self.records):
            raise ValueError("record stream does not match observations request")
        sequences = tuple(item.sequence for item in self.records)
        if any(left >= right for left, right in zip(sequences, sequences[1:])):
            raise ValueError("observation sequences must be unique and strictly increasing")
        if self.request.after_sequence is not None and sequences and sequences[0] <= self.request.after_sequence:
            raise ValueError("observation sequences must follow after_sequence")
        if type(self.truncated) is not bool:
            raise TypeError("truncated must be an exact boolean")
        if self.next_cursor is not None:
            _bounded_integer(self.next_cursor, "next_cursor", minimum=0, maximum=_MAX_SEQUENCE)
        if self.truncated != (self.next_cursor is not None):
            raise ValueError("next_cursor/truncated matrix invalid")
        if self.truncated and not self.records:
            raise ValueError("a truncated page must contain a record")
        if self.truncated and self.next_cursor != sequences[-1]:
            raise ValueError("next_cursor must equal the last record sequence")


__all__ = [
    "ChatTokenPayloadV1", "ChatTurnCompletedPayloadV1", "ChatTurnStartedPayloadV1",
    "DataRowWrittenPayloadV1", "DataScenarioCompletedPayloadV1",
    "DataStageGateEvaluatedPayloadV1", "EvaluationCaseScoredPayloadV1",
    "EvaluationCaseStartedPayloadV1", "EvaluationStageCompletedPayloadV1",
    "ObservationFamily", "ObservationKind", "ObservationPage", "ObservationRecordV1",
    "ObservationStreamRef", "ObservationsRequest", "PipelineStageCompletedPayloadV1",
    "PipelineStageStartedPayloadV1", "TrainingPhaseObservedPayloadV1",
]
