"""Closed, authority-free observation stream contracts (api-facade slice 1)."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import jsonschema
import pytest

from synaptic_tuner.api import v1
from synaptic_tuner.api.v1 import observations
from synaptic_tuner.api.v1.observations import (
    OBSERVATION_PAYLOAD_TYPES, ChatTokenPayloadV1, ChatTurnCompletedPayloadV1,
    ChatTurnStartedPayloadV1, DataRowWrittenPayloadV1, DataScenarioCompletedPayloadV1,
    DataStageGateEvaluatedPayloadV1, EvaluationCaseScoredPayloadV1,
    EvaluationCaseStartedPayloadV1, EvaluationStageCompletedPayloadV1,
    ObservationFamily, ObservationKind, ObservationPage, ObservationRecordV1,
    ObservationStreamRef, ObservationsRequest, PipelineStageCompletedPayloadV1,
    PipelineStageStartedPayloadV1, TrainingPhaseObservedPayloadV1,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "schemas/synaptic-observation-v1.schema.json").read_text(encoding="utf-8"))
DIGEST = "a" * 64
AT = "2026-09-17T12:00:04Z"

SAMPLE_PAYLOADS: dict[ObservationKind, object] = {
    ObservationKind.TRAINING_PHASE_OBSERVED: TrainingPhaseObservedPayloadV1("preparing"),
    ObservationKind.EVALUATION_CASE_STARTED: EvaluationCaseStartedPayloadV1("tool-choice#3"),
    ObservationKind.EVALUATION_CASE_SCORED: EvaluationCaseScoredPayloadV1("tool-choice#3", "passed", 0.82),
    ObservationKind.EVALUATION_STAGE_COMPLETED: EvaluationStageCompletedPayloadV1("scoring", 120),
    ObservationKind.CHAT_TURN_STARTED: ChatTurnStartedPayloadV1(1),
    ObservationKind.CHAT_TURN_COMPLETED: ChatTurnCompletedPayloadV1(1, DIGEST, 42),
    ObservationKind.CHAT_TOKEN: ChatTokenPayloadV1(1, 0, "Hello"),
    ObservationKind.DATA_ROW_WRITTEN: DataRowWrittenPayloadV1("scenario-a", 7),
    ObservationKind.DATA_STAGE_GATE_EVALUATED: DataStageGateEvaluatedPayloadV1("scenario-a", "rubric", True),
    ObservationKind.DATA_SCENARIO_COMPLETED: DataScenarioCompletedPayloadV1("scenario-a", 12),
    ObservationKind.PIPELINE_STAGE_STARTED: PipelineStageStartedPayloadV1("train", DIGEST),
    ObservationKind.PIPELINE_STAGE_COMPLETED: PipelineStageCompletedPayloadV1("train", DIGEST, "succeeded"),
}


def _stream(family: ObservationFamily = ObservationFamily.EVALUATION) -> ObservationStreamRef:
    return ObservationStreamRef(family, "acme", "ev-01J")


def _record(kind: ObservationKind, sequence: int = 41, stream: ObservationStreamRef | None = None) -> ObservationRecordV1:
    return ObservationRecordV1(
        "synaptic-observation/v1", stream or _stream(kind.family), sequence, AT, kind, SAMPLE_PAYLOADS[kind],
    )


def _scored(sequence: int, stream: ObservationStreamRef | None = None) -> ObservationRecordV1:
    return _record(ObservationKind.EVALUATION_CASE_SCORED, sequence, stream)


def test_root_exports_have_only_canonical_contract_identities() -> None:
    for name in observations.__all__:
        assert getattr(v1, name) is getattr(observations, name)
        assert name in v1.__all__


def test_kind_vocabulary_is_closed_and_bound_per_family() -> None:
    assert tuple(kind.value for kind in ObservationKind) == (
        "training_phase_observed",
        "evaluation_case_started", "evaluation_case_scored", "evaluation_stage_completed",
        "chat_turn_started", "chat_turn_completed", "chat_token",
        "data_row_written", "data_stage_gate_evaluated", "data_scenario_completed",
        "pipeline_stage_started", "pipeline_stage_completed",
    )
    assert tuple(family.value for family in ObservationFamily) == (
        "training", "evaluation", "chat", "data", "pipeline",
    )
    for kind in ObservationKind:
        assert kind.value.startswith(kind.family.value + "_")
        assert OBSERVATION_PAYLOAD_TYPES[kind].__name__.endswith("PayloadV1")
    assert set(OBSERVATION_PAYLOAD_TYPES) == set(ObservationKind)
    assert len({payload_type for payload_type in OBSERVATION_PAYLOAD_TYPES.values()}) == len(ObservationKind)
    assert set(SCHEMA["properties"]["kind"]["enum"]) == {kind.value for kind in ObservationKind}


def test_architecture_example_record_serializes_exactly() -> None:
    document = _scored(41).to_dict()
    assert document == {
        "schema_version": "synaptic-observation/v1",
        "stream": {"family": "evaluation", "project_ref": "acme", "entity_id": "ev-01J"},
        "sequence": 41, "occurred_at": AT,
        "kind": "evaluation_case_scored",
        "payload": {"case_ref": "tool-choice#3", "verdict": "passed", "score": 0.82},
    }


@pytest.mark.parametrize("kind", list(ObservationKind), ids=lambda kind: kind.value)
def test_every_kind_round_trips_through_exact_fields_and_the_closed_schema(kind: ObservationKind) -> None:
    record = _record(kind)
    document = record.to_dict()
    jsonschema.Draft202012Validator.check_schema(SCHEMA)
    jsonschema.validate(document, SCHEMA)
    assert ObservationRecordV1.from_dict(json.loads(json.dumps(document))) == record
    assert type(record.payload) is OBSERVATION_PAYLOAD_TYPES[kind]

    extra = json.loads(json.dumps(document))
    extra["payload"]["extra"] = "forbidden"
    with pytest.raises(ValueError, match="unknown fields"):
        ObservationRecordV1.from_dict(extra)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(extra, SCHEMA)

    missing = json.loads(json.dumps(document))
    dropped = sorted(missing["payload"])[0]
    del missing["payload"][dropped]
    with pytest.raises(ValueError, match="missing fields"):
        ObservationRecordV1.from_dict(missing)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(missing, SCHEMA)

    top = json.loads(json.dumps(document))
    top["authority"] = "none"
    with pytest.raises(ValueError, match="unknown fields"):
        ObservationRecordV1.from_dict(top)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(top, SCHEMA)


def test_unknown_kind_is_rejected_by_the_contract_and_the_schema() -> None:
    document = _scored(41).to_dict()
    document["kind"] = "evaluation_case_exploded"
    with pytest.raises(ValueError, match="unknown observation kind") as captured:
        ObservationRecordV1.from_dict(document)
    assert captured.value.__cause__ is None
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(document, SCHEMA)
    document["kind"] = 7
    with pytest.raises(TypeError, match="kind must be an exact string"):
        ObservationRecordV1.from_dict(document)
    with pytest.raises(TypeError, match="kind must be exact ObservationKind"):
        ObservationRecordV1("synaptic-observation/v1", _stream(), 1, AT, "evaluation_case_scored", SAMPLE_PAYLOADS[ObservationKind.EVALUATION_CASE_SCORED])  # type: ignore[arg-type]


def test_kind_must_belong_to_the_stream_family_and_select_its_payload_type() -> None:
    with pytest.raises(ValueError, match="does not belong to the stream family"):
        _record(ObservationKind.CHAT_TURN_STARTED, stream=_stream(ObservationFamily.EVALUATION))
    document = _scored(41).to_dict()
    document["stream"]["family"] = "chat"
    with pytest.raises(ValueError, match="does not belong to the stream family"):
        ObservationRecordV1.from_dict(document)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(document, SCHEMA)
    with pytest.raises(TypeError, match="payload must be exact EvaluationCaseScoredPayloadV1"):
        ObservationRecordV1(
            "synaptic-observation/v1", _stream(), 1, AT, ObservationKind.EVALUATION_CASE_SCORED,
            EvaluationCaseStartedPayloadV1("tool-choice#3"),
        )
    with pytest.raises(ValueError, match="unsupported observation schema version"):
        ObservationRecordV1("synaptic-observation/v2", _stream(), 1, AT, ObservationKind.EVALUATION_CASE_SCORED, SAMPLE_PAYLOADS[ObservationKind.EVALUATION_CASE_SCORED])
    document = _scored(41).to_dict()
    document["stream"]["family"] = "billing"
    with pytest.raises(ValueError, match="unknown observation family"):
        ObservationRecordV1.from_dict(document)


def test_payload_vocabularies_and_bounds_are_closed() -> None:
    with pytest.raises(ValueError, match="verdict must be one of"):
        EvaluationCaseScoredPayloadV1("case", "maybe", 0.5)
    with pytest.raises(ValueError, match="score must be finite"):
        EvaluationCaseScoredPayloadV1("case", "passed", float("nan"))
    with pytest.raises(TypeError, match="score must be a number"):
        EvaluationCaseScoredPayloadV1("case", "passed", True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="outcome must be one of"):
        PipelineStageCompletedPayloadV1("train", DIGEST, "completed")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        PipelineStageStartedPayloadV1("train", "A" * 64)
    with pytest.raises(ValueError, match="exceeds 4096"):
        ChatTokenPayloadV1(1, 0, "x" * 4097)
    with pytest.raises(ValueError, match="rows_written must be at least 1"):
        DataRowWrittenPayloadV1("scenario", 0)
    with pytest.raises(TypeError, match="passed must be an exact boolean"):
        DataStageGateEvaluatedPayloadV1("scenario", "gate", 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact integer"):
        ChatTurnStartedPayloadV1(True)  # type: ignore[arg-type]


def test_record_sequence_timestamp_and_stream_are_exact() -> None:
    with pytest.raises(ValueError, match="sequence must be an integer"):
        _scored(-1)
    with pytest.raises(ValueError, match="sequence must be an integer"):
        _scored(2**63)
    with pytest.raises(ValueError, match="sequence must be an integer"):
        _scored(True)  # type: ignore[arg-type]
    assert _scored(2**63 - 1).sequence == 2**63 - 1
    with pytest.raises(ValueError, match="occurred_at must be exact RFC3339"):
        replace(_scored(1), occurred_at="2026-09-17 12:00:04")
    with pytest.raises(TypeError, match="stream must be exact ObservationStreamRef"):
        replace(_scored(1), stream=("evaluation", "acme", "ev"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="project_ref is required"):
        ObservationStreamRef(ObservationFamily.EVALUATION, "", "ev")
    with pytest.raises(TypeError, match="family must be exact ObservationFamily"):
        ObservationStreamRef("evaluation", "acme", "ev")  # type: ignore[arg-type]


def test_observations_request_bounds_limit_and_after_sequence() -> None:
    request = ObservationsRequest(_stream())
    assert (request.after_sequence, request.limit) == (None, 100)
    assert ObservationsRequest(_stream(), 0, 200).limit == 200
    for limit in (0, 201, True, 1.0):
        with pytest.raises(ValueError, match="limit must be an integer from 1 through 200"):
            ObservationsRequest(_stream(), None, limit)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="after_sequence must be an integer"):
        ObservationsRequest(_stream(), -1)
    with pytest.raises(TypeError, match="stream must be exact ObservationStreamRef"):
        ObservationsRequest(("evaluation", "acme", "ev"))  # type: ignore[arg-type]


def test_page_next_cursor_and_truncated_must_agree() -> None:
    request = ObservationsRequest(_stream(), 40, 3)
    records = (_scored(41), _scored(42))
    page = ObservationPage(request, records, 42, True)
    assert (page.next_cursor, page.truncated) == (42, True)
    assert ObservationPage(request, records).truncated is False
    assert ObservationPage(request, ()).next_cursor is None
    with pytest.raises(ValueError, match="next_cursor/truncated matrix invalid"):
        ObservationPage(request, records, None, True)
    with pytest.raises(ValueError, match="next_cursor/truncated matrix invalid"):
        ObservationPage(request, records, 42, False)
    with pytest.raises(ValueError, match="a truncated page must contain a record"):
        ObservationPage(request, (), 42, True)
    with pytest.raises(ValueError, match="next_cursor must equal the last record sequence"):
        ObservationPage(request, records, 41, True)
    with pytest.raises(ValueError, match="next_cursor must equal the last record sequence"):
        ObservationPage(request, records, 43, True)
    with pytest.raises(TypeError, match="truncated must be an exact boolean"):
        ObservationPage(request, records, 42, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="next_cursor must be an integer"):
        ObservationPage(request, records, "42", True)  # type: ignore[arg-type]


def test_page_sequences_are_strictly_increasing_after_the_cursor_and_within_limit() -> None:
    request = ObservationsRequest(_stream(), 40, 2)
    with pytest.raises(ValueError, match="strictly increasing"):
        ObservationPage(request, (_scored(41), _scored(41)))
    with pytest.raises(ValueError, match="strictly increasing"):
        ObservationPage(request, (_scored(42), _scored(41)))
    with pytest.raises(ValueError, match="must follow after_sequence"):
        ObservationPage(request, (_scored(40), _scored(41)))
    with pytest.raises(ValueError, match="records exceed requested limit"):
        ObservationPage(request, (_scored(41), _scored(42), _scored(43)))
    with pytest.raises(ValueError, match="record stream does not match"):
        ObservationPage(request, (_scored(41, ObservationStreamRef(ObservationFamily.EVALUATION, "acme", "ev-02")),))
    with pytest.raises(TypeError, match="records must be an exact tuple"):
        ObservationPage(request, [_scored(41)])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="request must be exact ObservationsRequest"):
        ObservationPage(None, ())  # type: ignore[arg-type]
    unbounded = ObservationsRequest(_stream())
    assert ObservationPage(unbounded, (_scored(0), _scored(1))).records[0].sequence == 0


def test_observation_contracts_import_no_engine_provider_or_database_modules() -> None:
    script = f"""
import json, sys
sys.path.insert(0, {str(ROOT)!r})
import synaptic_tuner.api.v1.observations, synaptic_tuner.api.v1.usage
print(json.dumps(sorted(n for n in sys.modules if n == 'tuner' or n.startswith(('tuner.', 'modal', 'sqlite3', 'huggingface_hub', 'runpod')))))
"""
    completed = subprocess.run([sys.executable, "-I", "-c", script], cwd=ROOT, check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == []
    for name in ("observations.py", "usage.py"):
        path = ROOT / "synaptic_tuner/api/v1" / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        imports.update(alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
        assert imports.isdisjoint({"tuner", "modal", "sqlite3", "huggingface_hub", "runpod"})
