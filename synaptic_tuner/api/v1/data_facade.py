"""Dataset generation, improvement, listing and validation facade (contract only).

Location: ``synaptic_tuner/api/v1/data_facade.py``.

One generation or improvement run produces one dataset artifact and has its
own ``DataRunRef``. The ref is a distinct exact type, never a ``TrainingRunRef``
or an ``EvaluationRunRef``; the ``data`` store partition is the discriminator.
Per-scenario and per-row progress travels on the ``data`` observation stream
(``observations.py``), never in the durable record, so a run's durable event
count stays bounded regardless of how many rows it writes.

Every type here is a frozen slotted dataclass validating in ``__post_init__``
and round-tripping through ``exact_fields`` with canonical-JSON ``to_dict`` /
``from_dict``. ``DataAPI`` applies the ``RunsAPI._call`` discipline verbatim:
it rebuilds each input, hands the callback a detached copy, re-validates both
after the callback returns or raises, and rebuilds the result so nothing the
callback retains can alias what the caller receives. A result must bind the
request it answers.

Closed vocabularies: ``DataRunState`` (``partially_succeeded`` is a terminal
state: a verified artifact exists but holds fewer rows than requested; a
``cancelled`` run likewise lists the rows it produced), ``DataMode``,
``DataOperationCode`` and ``ValidationFindingCode``. The two artifact roles a
run may publish are ``dataset_jsonl`` (the homogeneous data file, every line
a row) and ``dataset_metadata`` (the sidecar that replaced the former
``_meta`` header row). ``usage`` is present only when measured. Read-only
``datasets`` and ``validate`` take no run and page with the settled cursor
contract; an implementation refuses an undecodable cursor with
``cursor_invalid``.

Registered in the lazy export table of ``synaptic_tuner/api/v1/__init__.py``
and in both import-closure gates. Imports nothing from ``tuner.*`` or
``SynthChat``; the reference implementation lives in ``api/v1/reference/data.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol

from ._contract import contract_digest, digest_text, exact_fields, required_text
from ._timestamps import require_rfc3339
from .observations import (
    ObservationFamily,
    ObservationPage,
    ObservationRecordV1,
    ObservationStreamRef,
    ObservationsRequest,
)
from .results import VerifiedArtifact
from .training_facade import AuthorizationRequirement
from .usage import UsageAvailability, UsageRecordV1


DATA_PLAN_SCHEMA_VERSION = "synaptic-dataset-plan/v1"
DATA_RESULT_SCHEMA_VERSION = "synaptic-dataset-result/v1"

DATASET_JSONL_ROLE = "dataset_jsonl"
DATASET_METADATA_ROLE = "dataset_metadata"
DATASET_ARTIFACT_ROLES = frozenset({DATASET_JSONL_ROLE, DATASET_METADATA_ROLE})

_MAX_LIST_LIMIT = 100
_MAX_COUNT = 2**63 - 1


# --- exact-value helpers ------------------------------------------------------


def _text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    return required_text(value, name)


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact boolean")
    return value


def _count(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if not minimum <= value <= _MAX_COUNT:
        raise ValueError(f"{name} must be an integer from {minimum} through {_MAX_COUNT}")
    return value


def _exact_list(value: object, name: str) -> tuple[object, ...]:
    if type(value) is not list:
        raise TypeError(f"{name} must be an exact array")
    return tuple(value)


def _text_set(value: object, name: str, *, minimum: int = 0) -> tuple[str, ...]:
    """Unique, ascending exact strings; the canonical form of a set of refs."""
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    items = tuple(_text(item, name) for item in value)
    if len(items) != len(set(items)):
        raise ValueError(f"{name} must be unique")
    if len(items) < minimum:
        raise ValueError(f"{name} requires at least {minimum} entry")
    return tuple(sorted(items))


def _ascii_cursor(value: str | None, name: str = "cursor") -> str | None:
    if value is None:
        return None
    value = _text(value, name)
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError(f"{name} must be ASCII") from None
    if len(encoded) > 256:
        raise ValueError(f"{name} exceeds 256 bytes")
    return value


def _artifacts(value: object) -> tuple[VerifiedArtifact, ...]:
    if type(value) is not tuple or any(type(item) is not VerifiedArtifact for item in value):
        raise TypeError("artifacts must be an exact tuple of VerifiedArtifact values")
    artifacts = tuple(VerifiedArtifact.from_dict(item.to_dict()) for item in value)
    roles = tuple(item.role for item in artifacts)
    if len(roles) != len(set(roles)):
        raise ValueError("artifact roles must be unique")
    if any(role not in DATASET_ARTIFACT_ROLES for role in roles):
        raise ValueError(f"artifact roles must be one of: {', '.join(sorted(DATASET_ARTIFACT_ROLES))}")
    return tuple(sorted(artifacts, key=lambda item: item.role))


def _artifact_list(artifacts: tuple[VerifiedArtifact, ...]) -> list[dict[str, object]]:
    return [item.to_dict() for item in artifacts]


def _parse_artifact_list(value: object) -> tuple[VerifiedArtifact, ...]:
    return tuple(
        VerifiedArtifact.from_dict(item)  # type: ignore[arg-type]
        for item in _exact_list(value, "artifacts")
    )


def _parse_enum(enum_type: type, value: object, name: str):
    value = _text(value, name)
    try:
        return enum_type(value)
    except ValueError:
        raise ValueError(f"unknown {name}") from None


def _timestamp(value: str, name: str) -> datetime:
    value = require_rfc3339(_text(value, name), name)
    return datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)


def _exact_tuple_of(value: object, item_type: type, name: str) -> tuple:
    if type(value) is not tuple or any(type(item) is not item_type for item in value):
        raise TypeError(f"{name} must be an exact tuple of {item_type.__name__} values")
    return tuple(item_type.from_dict(item.to_dict()) for item in value)


# --- identities and closed vocabularies ----------------------------------------


@dataclass(frozen=True, slots=True)
class DataRunRef:
    """Identity of one dataset run; never interchangeable with any other run ref."""

    run_id: str
    project_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", required_text(self.run_id, "run_id"))
        object.__setattr__(self, "project_ref", required_text(self.project_ref, "project_ref"))

    def to_dict(self) -> dict[str, object]:
        return {"run_id": self.run_id, "project_ref": self.project_ref}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "DataRunRef":
        value = exact_fields(value, frozenset({"run_id", "project_ref"}), "data_run_ref")
        return cls(run_id=value["run_id"], project_ref=value["project_ref"])  # type: ignore[arg-type]


class DataRunState(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIALLY_SUCCEEDED = "partially_succeeded"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in TERMINAL_DATA_STATES


TERMINAL_DATA_STATES = frozenset({
    DataRunState.SUCCEEDED, DataRunState.PARTIALLY_SUCCEEDED,
    DataRunState.FAILED, DataRunState.CANCELLED,
})


class DataMode(str, Enum):
    """What the run does: generate new rows from scenarios, or improve an existing dataset."""

    GENERATE = "generate"
    IMPROVE = "improve"


class DataOperationCode(str, Enum):
    RUN_MISSING = "run_missing"
    CURSOR_INVALID = "cursor_invalid"
    SCENARIO_INVALID = "scenario_invalid"
    MANIFEST_INVALID = "manifest_invalid"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    BACKEND_UNMETERED = "backend_unmetered"
    SPEND_INDETERMINATE = "spend_indeterminate"
    OUTPUT_CONFLICT = "output_conflict"
    CANCEL_INELIGIBLE = "cancel_ineligible"
    DATASET_MISSING = "dataset_missing"
    STATE_CONFLICT = "state_conflict"
    INTEGRITY_ERROR = "integrity_error"


class DataOperationError(ValueError):
    def __init__(self, code: DataOperationCode) -> None:
        if type(code) is not DataOperationCode:
            raise TypeError("code must be exact DataOperationCode")
        self.code = code
        super().__init__(code.value)


class ValidationFindingCode(str, Enum):
    """Structural defects a read-only dataset validation can report per line."""

    MALFORMED_JSON = "malformed_json"
    ROW_NOT_OBJECT = "row_not_object"
    LEGACY_METADATA_ROW = "legacy_metadata_row"
    CONVERSATIONS_MISSING = "conversations_missing"
    CONVERSATIONS_INVALID = "conversations_invalid"


# --- scenarios ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DataScenarioTarget:
    """How many rows one scenario should contribute."""

    scenario_ref: str
    rows: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_ref", _text(self.scenario_ref, "scenario_ref"))
        _count(self.rows, "rows", minimum=1)

    def to_dict(self) -> dict[str, object]:
        return {"scenario_ref": self.scenario_ref, "rows": self.rows}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "DataScenarioTarget":
        value = exact_fields(value, frozenset({"scenario_ref", "rows"}), "scenario_target")
        return cls(value["scenario_ref"], value["rows"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class DataScenarioOutcome:
    """How many rows one scenario was asked for and how many it wrote."""

    scenario_ref: str
    rows_requested: int
    rows_written: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_ref", _text(self.scenario_ref, "scenario_ref"))
        _count(self.rows_requested, "rows_requested", minimum=1)
        _count(self.rows_written, "rows_written")
        if self.rows_written > self.rows_requested:
            raise ValueError("rows_written must not exceed rows_requested")

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_ref": self.scenario_ref,
            "rows_requested": self.rows_requested,
            "rows_written": self.rows_written,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "DataScenarioOutcome":
        value = exact_fields(
            value, frozenset({"scenario_ref", "rows_requested", "rows_written"}), "scenario_outcome"
        )
        return cls(value["scenario_ref"], value["rows_requested"], value["rows_written"])  # type: ignore[arg-type]


def _scenario_targets(value: object, name: str) -> tuple[DataScenarioTarget, ...]:
    targets = _exact_tuple_of(value, DataScenarioTarget, name)
    refs = tuple(item.scenario_ref for item in targets)
    if len(refs) != len(set(refs)):
        raise ValueError(f"{name} scenario refs must be unique")
    return tuple(sorted(targets, key=lambda item: item.scenario_ref))


def _scenario_outcomes(value: object, *, rows_requested: int, rows_written: int) -> tuple[DataScenarioOutcome, ...]:
    outcomes = _exact_tuple_of(value, DataScenarioOutcome, "scenarios")
    refs = tuple(item.scenario_ref for item in outcomes)
    if len(refs) != len(set(refs)):
        raise ValueError("scenarios scenario refs must be unique")
    if not outcomes:
        raise ValueError("scenarios requires at least 1 entry")
    if sum(item.rows_requested for item in outcomes) != rows_requested:
        raise ValueError("scenario rows_requested must sum to rows_requested")
    if sum(item.rows_written for item in outcomes) != rows_written:
        raise ValueError("scenario rows_written must sum to rows_written")
    return tuple(sorted(outcomes, key=lambda item: item.scenario_ref))


def _parse_targets(value: object, name: str) -> tuple[DataScenarioTarget, ...]:
    return tuple(DataScenarioTarget.from_dict(item) for item in _exact_list(value, name))  # type: ignore[arg-type]


def _parse_outcomes(value: object) -> tuple[DataScenarioOutcome, ...]:
    return tuple(DataScenarioOutcome.from_dict(item) for item in _exact_list(value, "scenarios"))  # type: ignore[arg-type]


# --- request, plan, preflight, start -------------------------------------------


_REQUEST_FIELDS = frozenset({
    "request_id", "project_ref", "mode", "backend", "model", "output_ref", "scenarios",
    "input_dataset_ref", "rubric_refs", "max_iterations", "tags",
})


@dataclass(frozen=True, slots=True)
class DataRequest:
    """One backend, one model, one output and either scenario targets or an input dataset.

    ``generate`` names at least one scenario target and no input dataset;
    ``improve`` names an input dataset and at least one rubric and no scenario
    targets (the plan derives one target per input dataset). ``output_ref`` is
    supplied by the host; no facade constructs a path from a project context.
    ``backend`` is an open name at the contract level; an implementation
    refuses an unknown one with ``backend_unavailable`` and an unmetered paid
    one with ``backend_unmetered``. Secrets never appear here.
    """

    request_id: str
    project_ref: str
    mode: DataMode
    backend: str
    model: str
    output_ref: str
    scenarios: tuple[DataScenarioTarget, ...] = ()
    input_dataset_ref: str | None = None
    rubric_refs: tuple[str, ...] = ()
    max_iterations: int = 1
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _text(self.request_id, "request_id"))
        object.__setattr__(self, "project_ref", _text(self.project_ref, "project_ref"))
        if type(self.mode) is not DataMode:
            raise TypeError("mode must be exact DataMode")
        object.__setattr__(self, "backend", _text(self.backend, "backend"))
        object.__setattr__(self, "model", _text(self.model, "model"))
        object.__setattr__(self, "output_ref", _text(self.output_ref, "output_ref"))
        object.__setattr__(self, "scenarios", _scenario_targets(self.scenarios, "scenarios"))
        object.__setattr__(self, "input_dataset_ref", _optional_text(self.input_dataset_ref, "input_dataset_ref"))
        object.__setattr__(self, "rubric_refs", _text_set(self.rubric_refs, "rubric_refs"))
        _count(self.max_iterations, "max_iterations", minimum=1)
        object.__setattr__(self, "tags", _text_set(self.tags, "tags"))
        if self.mode is DataMode.GENERATE:
            if not self.scenarios:
                raise ValueError("generate requires at least 1 scenario target")
            if self.input_dataset_ref is not None:
                raise ValueError("generate takes no input_dataset_ref")
        else:
            if self.scenarios:
                raise ValueError("improve takes no scenario targets")
            if self.input_dataset_ref is None:
                raise ValueError("improve requires input_dataset_ref")
            if not self.rubric_refs:
                raise ValueError("improve requires at least 1 rubric ref")

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "project_ref": self.project_ref,
            "mode": self.mode.value,
            "backend": self.backend,
            "model": self.model,
            "output_ref": self.output_ref,
            "scenarios": [item.to_dict() for item in self.scenarios],
            "input_dataset_ref": self.input_dataset_ref,
            "rubric_refs": list(self.rubric_refs),
            "max_iterations": self.max_iterations,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "DataRequest":
        value = exact_fields(value, _REQUEST_FIELDS, "data_request")
        return cls(
            value["request_id"],  # type: ignore[arg-type]
            value["project_ref"],  # type: ignore[arg-type]
            _parse_enum(DataMode, value["mode"], "mode"),
            value["backend"],  # type: ignore[arg-type]
            value["model"],  # type: ignore[arg-type]
            value["output_ref"],  # type: ignore[arg-type]
            _parse_targets(value["scenarios"], "scenarios"),
            value["input_dataset_ref"],  # type: ignore[arg-type]
            _exact_list(value["rubric_refs"], "rubric_refs"),  # type: ignore[arg-type]
            value["max_iterations"],  # type: ignore[arg-type]
            _exact_list(value["tags"], "tags"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class DataPlan:
    """A resolved request: the scenario targets it yields, their row total and source digest.

    For ``generate`` the targets are the request's own; for ``improve`` the
    implementation derives one target named by the input dataset holding as
    many rows as the dataset. ``plan_fingerprint`` is derived, never stored;
    ``DataPreflight`` binds to it.
    """

    schema_version: str
    request: DataRequest
    scenarios: tuple[DataScenarioTarget, ...]
    rows_requested: int
    source_digest: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != DATA_PLAN_SCHEMA_VERSION:
            raise ValueError("unsupported dataset plan schema version")
        if type(self.request) is not DataRequest:
            raise TypeError("request must be exact DataRequest")
        object.__setattr__(self, "request", DataRequest.from_dict(self.request.to_dict()))
        scenarios = _scenario_targets(self.scenarios, "scenarios")
        if not scenarios:
            raise ValueError("scenarios requires at least 1 entry")
        object.__setattr__(self, "scenarios", scenarios)
        _count(self.rows_requested, "rows_requested", minimum=1)
        if sum(item.rows for item in scenarios) != self.rows_requested:
            raise ValueError("scenario rows must sum to rows_requested")
        if self.request.mode is DataMode.GENERATE and scenarios != self.request.scenarios:
            raise ValueError("generate plan scenarios must equal the request targets")
        object.__setattr__(self, "source_digest", digest_text(_text(self.source_digest, "source_digest"), "source_digest"))

    @property
    def plan_fingerprint(self) -> str:
        return contract_digest(DATA_PLAN_SCHEMA_VERSION, self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "request": self.request.to_dict(),
            "scenarios": [item.to_dict() for item in self.scenarios],
            "rows_requested": self.rows_requested,
            "source_digest": self.source_digest,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "DataPlan":
        value = exact_fields(
            value,
            frozenset({"schema_version", "request", "scenarios", "rows_requested", "source_digest"}),
            "dataset_plan",
        )
        return cls(
            _text(value["schema_version"], "schema_version"),
            DataRequest.from_dict(value["request"]),  # type: ignore[arg-type]
            _parse_targets(value["scenarios"], "scenarios"),
            value["rows_requested"],  # type: ignore[arg-type]
            value["source_digest"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class DataPreflight:
    """Readiness of one exact plan, with the authorization a start would need."""

    plan_fingerprint: str
    ready: bool
    checked_at: str
    expires_at: str
    authorization: tuple[AuthorizationRequirement, ...] = ()
    diagnostic_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_fingerprint", digest_text(_text(self.plan_fingerprint, "plan_fingerprint"), "plan_fingerprint"))
        _bool(self.ready, "ready")
        if _timestamp(self.expires_at, "expires_at") <= _timestamp(self.checked_at, "checked_at"):
            raise ValueError("expires_at must be later than checked_at")
        if type(self.authorization) is not tuple or any(
            type(item) is not AuthorizationRequirement for item in self.authorization
        ):
            raise TypeError("authorization must be an exact tuple of AuthorizationRequirement values")
        authorization = tuple(AuthorizationRequirement.from_dict(item.to_dict()) for item in self.authorization)
        operations = tuple(item.operation for item in authorization)
        if len(operations) != len(set(operations)):
            raise ValueError("authorization operations must be unique")
        codes = _text_set(self.diagnostic_codes, "diagnostic_codes")
        if not self.ready and not codes:
            raise ValueError("not-ready preflight requires a diagnostic code")
        object.__setattr__(self, "authorization", tuple(sorted(authorization, key=lambda item: item.operation)))
        object.__setattr__(self, "diagnostic_codes", codes)

    def binds(self, plan: DataPlan) -> bool:
        if type(plan) is not DataPlan:
            raise TypeError("plan must be exact DataPlan")
        return self.plan_fingerprint == plan.plan_fingerprint

    def is_expired(self, now: str) -> bool:
        return _timestamp(now, "now") >= _timestamp(self.expires_at, "expires_at")

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_fingerprint": self.plan_fingerprint,
            "ready": self.ready,
            "checked_at": self.checked_at,
            "expires_at": self.expires_at,
            "authorization": {
                item.operation: {
                    "paid_effect": item.paid_effect,
                    "maximum_cost_minor_units": item.maximum_cost_minor_units,
                    "currency": item.currency,
                }
                for item in self.authorization
            },
            "diagnostic_codes": list(self.diagnostic_codes),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "DataPreflight":
        value = exact_fields(
            value,
            frozenset({
                "plan_fingerprint", "ready", "checked_at", "expires_at",
                "authorization", "diagnostic_codes",
            }),
            "dataset_preflight",
        )
        authorization = value["authorization"]
        if type(authorization) is not dict:
            raise TypeError("authorization must be an exact operation-keyed object")
        operations = tuple(dict.keys(authorization))
        if any(type(operation) is not str for operation in operations):
            raise TypeError("authorization field names must be exact strings")
        requirements = []
        for operation in sorted(operations):
            descriptor = exact_fields(
                dict.__getitem__(authorization, operation),
                frozenset({"paid_effect", "maximum_cost_minor_units", "currency"}),
                "authorization_requirement",
            )
            requirements.append(AuthorizationRequirement.from_dict({"operation": operation, **descriptor}))
        return cls(
            value["plan_fingerprint"],  # type: ignore[arg-type]
            value["ready"],  # type: ignore[arg-type]
            value["checked_at"],  # type: ignore[arg-type]
            value["expires_at"],  # type: ignore[arg-type]
            tuple(requirements),
            _exact_list(value["diagnostic_codes"], "diagnostic_codes"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class DataStart:
    run: DataRunRef
    accepted: bool

    def __post_init__(self) -> None:
        if type(self.run) is not DataRunRef:
            raise TypeError("run must be exact DataRunRef")
        object.__setattr__(self, "run", DataRunRef.from_dict(self.run.to_dict()))
        _bool(self.accepted, "accepted")

    def to_dict(self) -> dict[str, object]:
        return {"run": self.run.to_dict(), "accepted": self.accepted}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "DataStart":
        value = exact_fields(value, frozenset({"run", "accepted"}), "dataset_start")
        return cls(DataRunRef.from_dict(value["run"]), value["accepted"])  # type: ignore[arg-type]


# --- outcome, paging, result ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class DataOutcome:
    """The durable state view of one run.

    Artifacts are permitted on any state: a ``partially_succeeded`` or
    ``cancelled`` run lists the verified artifact holding the rows it did
    produce.
    """

    run: DataRunRef
    state: DataRunState
    rows_requested: int
    rows_written: int
    scenarios: tuple[DataScenarioOutcome, ...]
    artifacts: tuple[VerifiedArtifact, ...] = ()
    diagnostic_code: str | None = None

    def __post_init__(self) -> None:
        if type(self.run) is not DataRunRef or type(self.state) is not DataRunState:
            raise TypeError("run/state have invalid types")
        object.__setattr__(self, "run", DataRunRef.from_dict(self.run.to_dict()))
        _count(self.rows_requested, "rows_requested", minimum=1)
        _count(self.rows_written, "rows_written")
        if self.rows_written > self.rows_requested:
            raise ValueError("rows_written must not exceed rows_requested")
        object.__setattr__(
            self, "scenarios",
            _scenario_outcomes(self.scenarios, rows_requested=self.rows_requested, rows_written=self.rows_written),
        )
        object.__setattr__(self, "artifacts", _artifacts(self.artifacts))
        object.__setattr__(self, "diagnostic_code", _optional_text(self.diagnostic_code, "diagnostic_code"))

    def to_dict(self) -> dict[str, object]:
        return {
            "run": self.run.to_dict(),
            "state": self.state.value,
            "rows_requested": self.rows_requested,
            "rows_written": self.rows_written,
            "scenarios": [item.to_dict() for item in self.scenarios],
            "artifacts": _artifact_list(self.artifacts),
            "diagnostic_code": self.diagnostic_code,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "DataOutcome":
        value = exact_fields(
            value,
            frozenset({"run", "state", "rows_requested", "rows_written", "scenarios", "artifacts", "diagnostic_code"}),
            "dataset_outcome",
        )
        return cls(
            DataRunRef.from_dict(value["run"]),  # type: ignore[arg-type]
            _parse_enum(DataRunState, value["state"], "state"),
            value["rows_requested"],  # type: ignore[arg-type]
            value["rows_written"],  # type: ignore[arg-type]
            _parse_outcomes(value["scenarios"]),
            _parse_artifact_list(value["artifacts"]),
            value["diagnostic_code"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class DataListRequest:
    project_ref: str
    cursor: str | None = None
    limit: int = _MAX_LIST_LIMIT

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_ref", _text(self.project_ref, "project_ref"))
        object.__setattr__(self, "cursor", _ascii_cursor(self.cursor))
        if type(self.limit) is not int or not 1 <= self.limit <= _MAX_LIST_LIMIT:
            raise ValueError(f"limit must be an integer from 1 through {_MAX_LIST_LIMIT}")


@dataclass(frozen=True, slots=True)
class DataPage:
    request: DataListRequest
    outcomes: tuple[DataOutcome, ...]
    next_cursor: str | None = None
    truncated: bool = False

    def __post_init__(self) -> None:
        if type(self.request) is not DataListRequest:
            raise TypeError("request must be exact DataListRequest")
        if type(self.outcomes) is not tuple or any(type(item) is not DataOutcome for item in self.outcomes):
            raise TypeError("outcomes must be an exact tuple of DataOutcome")
        if len(self.outcomes) > self.request.limit:
            raise ValueError("outcomes exceed requested limit")
        if any(item.run.project_ref != self.request.project_ref for item in self.outcomes):
            raise ValueError("outcome project does not match list request")
        _bool(self.truncated, "truncated")
        object.__setattr__(self, "next_cursor", _ascii_cursor(self.next_cursor, "next_cursor"))
        if self.truncated != (self.next_cursor is not None):
            raise ValueError("next_cursor/truncated matrix invalid")
        if self.truncated and not self.outcomes:
            raise ValueError("a truncated page must contain an outcome")


# --- read-only dataset queries ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class DatasetDescriptor:
    """One dataset file a host directory holds, with its metadata sidecar when present."""

    dataset_ref: str
    size_bytes: int
    metadata_ref: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_ref", _text(self.dataset_ref, "dataset_ref"))
        _count(self.size_bytes, "size_bytes")
        object.__setattr__(self, "metadata_ref", _optional_text(self.metadata_ref, "metadata_ref"))

    def to_dict(self) -> dict[str, object]:
        return {"dataset_ref": self.dataset_ref, "size_bytes": self.size_bytes, "metadata_ref": self.metadata_ref}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "DatasetDescriptor":
        value = exact_fields(value, frozenset({"dataset_ref", "size_bytes", "metadata_ref"}), "dataset_descriptor")
        return cls(value["dataset_ref"], value["size_bytes"], value["metadata_ref"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class DatasetListRequest:
    """List the dataset files under a host-supplied directory; takes no run."""

    project_ref: str
    directory_ref: str
    cursor: str | None = None
    limit: int = _MAX_LIST_LIMIT

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_ref", _text(self.project_ref, "project_ref"))
        object.__setattr__(self, "directory_ref", _text(self.directory_ref, "directory_ref"))
        object.__setattr__(self, "cursor", _ascii_cursor(self.cursor))
        if type(self.limit) is not int or not 1 <= self.limit <= _MAX_LIST_LIMIT:
            raise ValueError(f"limit must be an integer from 1 through {_MAX_LIST_LIMIT}")


@dataclass(frozen=True, slots=True)
class DatasetPage:
    request: DatasetListRequest
    datasets: tuple[DatasetDescriptor, ...]
    next_cursor: str | None = None
    truncated: bool = False

    def __post_init__(self) -> None:
        if type(self.request) is not DatasetListRequest:
            raise TypeError("request must be exact DatasetListRequest")
        object.__setattr__(self, "datasets", _exact_tuple_of(self.datasets, DatasetDescriptor, "datasets"))
        if len(self.datasets) > self.request.limit:
            raise ValueError("datasets exceed requested limit")
        refs = tuple(item.dataset_ref for item in self.datasets)
        if any(left >= right for left, right in zip(refs, refs[1:])):
            raise ValueError("dataset refs must be unique and strictly increasing")
        _bool(self.truncated, "truncated")
        object.__setattr__(self, "next_cursor", _ascii_cursor(self.next_cursor, "next_cursor"))
        if self.truncated != (self.next_cursor is not None):
            raise ValueError("next_cursor/truncated matrix invalid")
        if self.truncated and not self.datasets:
            raise ValueError("a truncated page must contain a dataset")


@dataclass(frozen=True, slots=True)
class ValidationFinding:
    line_number: int
    code: ValidationFindingCode

    def __post_init__(self) -> None:
        _count(self.line_number, "line_number", minimum=1)
        if type(self.code) is not ValidationFindingCode:
            raise TypeError("code must be exact ValidationFindingCode")

    def to_dict(self) -> dict[str, object]:
        return {"line_number": self.line_number, "code": self.code.value}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ValidationFinding":
        value = exact_fields(value, frozenset({"line_number", "code"}), "validation_finding")
        return cls(value["line_number"], _parse_enum(ValidationFindingCode, value["code"], "code"))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class DatasetValidateRequest:
    """Structurally validate one page of rows of a dataset file; takes no run and no LLM."""

    project_ref: str
    dataset_ref: str
    cursor: str | None = None
    limit: int = _MAX_LIST_LIMIT

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_ref", _text(self.project_ref, "project_ref"))
        object.__setattr__(self, "dataset_ref", _text(self.dataset_ref, "dataset_ref"))
        object.__setattr__(self, "cursor", _ascii_cursor(self.cursor))
        if type(self.limit) is not int or not 1 <= self.limit <= _MAX_LIST_LIMIT:
            raise ValueError(f"limit must be an integer from 1 through {_MAX_LIST_LIMIT}")


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """One page of structural findings; every invalid row yields exactly one finding."""

    request: DatasetValidateRequest
    rows_checked: int
    rows_valid: int
    findings: tuple[ValidationFinding, ...] = ()
    next_cursor: str | None = None
    truncated: bool = False

    def __post_init__(self) -> None:
        if type(self.request) is not DatasetValidateRequest:
            raise TypeError("request must be exact DatasetValidateRequest")
        _count(self.rows_checked, "rows_checked")
        _count(self.rows_valid, "rows_valid")
        if self.rows_checked > self.request.limit:
            raise ValueError("rows_checked exceed requested limit")
        if self.rows_valid > self.rows_checked:
            raise ValueError("rows_valid must not exceed rows_checked")
        object.__setattr__(self, "findings", _exact_tuple_of(self.findings, ValidationFinding, "findings"))
        lines = tuple(item.line_number for item in self.findings)
        if any(left >= right for left, right in zip(lines, lines[1:])):
            raise ValueError("finding line numbers must be unique and strictly increasing")
        if len(self.findings) != self.rows_checked - self.rows_valid:
            raise ValueError("findings must number exactly the invalid rows")
        _bool(self.truncated, "truncated")
        object.__setattr__(self, "next_cursor", _ascii_cursor(self.next_cursor, "next_cursor"))
        if self.truncated != (self.next_cursor is not None):
            raise ValueError("next_cursor/truncated matrix invalid")
        if self.truncated and self.rows_checked == 0:
            raise ValueError("a truncated report must have checked a row")


_RESULT_FIELDS = frozenset({
    "schema_version", "run", "state", "mode", "backend", "model", "rows_requested",
    "rows_written", "scenarios", "artifacts", "diagnostic_code",
})


@dataclass(frozen=True, slots=True)
class DataResult:
    """The public record of one run: counts, per-scenario counts, verified artifacts.

    Row content never enters this record. ``usage`` is present only when the
    backend measured it; an ``unavailable`` usage is refused here and the field
    is omitted from the canonical document rather than published as null.
    """

    schema_version: str
    run: DataRunRef
    state: DataRunState
    mode: DataMode
    backend: str
    model: str
    rows_requested: int
    rows_written: int
    scenarios: tuple[DataScenarioOutcome, ...]
    artifacts: tuple[VerifiedArtifact, ...] = ()
    diagnostic_code: str | None = None
    usage: UsageRecordV1 | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != DATA_RESULT_SCHEMA_VERSION:
            raise ValueError("unsupported dataset result schema version")
        if type(self.run) is not DataRunRef or type(self.state) is not DataRunState:
            raise TypeError("run/state have invalid types")
        object.__setattr__(self, "run", DataRunRef.from_dict(self.run.to_dict()))
        if type(self.mode) is not DataMode:
            raise TypeError("mode must be exact DataMode")
        object.__setattr__(self, "backend", _text(self.backend, "backend"))
        object.__setattr__(self, "model", _text(self.model, "model"))
        _count(self.rows_requested, "rows_requested", minimum=1)
        _count(self.rows_written, "rows_written")
        if self.rows_written > self.rows_requested:
            raise ValueError("rows_written must not exceed rows_requested")
        object.__setattr__(
            self, "scenarios",
            _scenario_outcomes(self.scenarios, rows_requested=self.rows_requested, rows_written=self.rows_written),
        )
        object.__setattr__(self, "artifacts", _artifacts(self.artifacts))
        object.__setattr__(self, "diagnostic_code", _optional_text(self.diagnostic_code, "diagnostic_code"))
        if self.usage is not None:
            if type(self.usage) is not UsageRecordV1:
                raise TypeError("usage must be exact UsageRecordV1 or None")
            if self.usage.availability is not UsageAvailability.MEASURED:
                raise ValueError("unavailable usage must be omitted, not published")
            object.__setattr__(self, "usage", UsageRecordV1.from_dict(self.usage.to_dict()))

    def outcome(self) -> DataOutcome:
        """The state view a ``show`` returns for this record."""
        return DataOutcome(
            self.run, self.state, self.rows_requested, self.rows_written,
            self.scenarios, self.artifacts, self.diagnostic_code,
        )

    def to_dict(self) -> dict[str, object]:
        document: dict[str, object] = {
            "schema_version": self.schema_version,
            "run": self.run.to_dict(),
            "state": self.state.value,
            "mode": self.mode.value,
            "backend": self.backend,
            "model": self.model,
            "rows_requested": self.rows_requested,
            "rows_written": self.rows_written,
            "scenarios": [item.to_dict() for item in self.scenarios],
            "artifacts": _artifact_list(self.artifacts),
            "diagnostic_code": self.diagnostic_code,
        }
        if self.usage is not None:
            document["usage"] = self.usage.to_dict()
        return document

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "DataResult":
        if type(value) is not dict:
            raise TypeError("dataset_result must be an exact object")
        keys = tuple(dict.keys(value))
        if any(type(key) is not str for key in keys):
            raise TypeError("dataset_result field names must be exact strings")
        expected = _RESULT_FIELDS | {"usage"} if "usage" in keys else _RESULT_FIELDS
        value = exact_fields(value, expected, "dataset_result")
        usage = value.get("usage")
        if "usage" in value and usage is None:
            raise ValueError("usage must be omitted when unavailable, not null")
        return cls(
            _text(value["schema_version"], "schema_version"),
            DataRunRef.from_dict(value["run"]),  # type: ignore[arg-type]
            _parse_enum(DataRunState, value["state"], "state"),
            _parse_enum(DataMode, value["mode"], "mode"),
            value["backend"],  # type: ignore[arg-type]
            value["model"],  # type: ignore[arg-type]
            value["rows_requested"],  # type: ignore[arg-type]
            value["rows_written"],  # type: ignore[arg-type]
            _parse_outcomes(value["scenarios"]),
            _parse_artifact_list(value["artifacts"]),
            value["diagnostic_code"],  # type: ignore[arg-type]
            None if usage is None else UsageRecordV1.from_dict(usage),  # type: ignore[arg-type]
        )


# --- operations and facade -------------------------------------------------------


class DataOperations(Protocol):
    def plan(self, request: DataRequest) -> DataPlan: ...
    def preflight(self, plan: DataPlan) -> DataPreflight: ...
    def start(self, plan: DataPlan) -> DataStart: ...
    def show(self, run: DataRunRef) -> DataOutcome: ...
    def cancel(self, run: DataRunRef, reason: str) -> DataOutcome: ...
    def list(self, request: DataListRequest) -> DataPage: ...
    def datasets(self, request: DatasetListRequest) -> DatasetPage: ...
    def validate(self, request: DatasetValidateRequest) -> ValidationReport: ...
    def observations(self, request: ObservationsRequest) -> ObservationPage: ...


class DataAPI:
    """Public data facade over host-composed ``DataOperations``.

    Every verb rebuilds its input, presents a detached copy to the callback,
    re-validates the original and the copy after return or raise, and rebuilds
    the callback's result so nothing the callback retains can alias the values
    handed back to the caller.
    """

    __slots__ = ("_operations",)

    def __init__(self, operations: DataOperations) -> None:
        self._operations = operations

    # -- input rebuilders (one per accepted input type) --------------------------

    @staticmethod
    def _run(value: DataRunRef) -> DataRunRef:
        if type(value) is not DataRunRef:
            raise TypeError("run must be exact DataRunRef")
        return DataRunRef.from_dict(value.to_dict())

    @staticmethod
    def _request(value: DataRequest) -> DataRequest:
        if type(value) is not DataRequest:
            raise TypeError("request must be exact DataRequest")
        return DataRequest.from_dict(value.to_dict())

    @staticmethod
    def _plan(value: DataPlan) -> DataPlan:
        if type(value) is not DataPlan:
            raise TypeError("plan must be exact DataPlan")
        return DataPlan.from_dict(value.to_dict())

    @staticmethod
    def _list_request(value: DataListRequest) -> DataListRequest:
        if type(value) is not DataListRequest:
            raise TypeError("request must be exact DataListRequest")
        return DataListRequest(value.project_ref, value.cursor, value.limit)

    @staticmethod
    def _dataset_list_request(value: DatasetListRequest) -> DatasetListRequest:
        if type(value) is not DatasetListRequest:
            raise TypeError("request must be exact DatasetListRequest")
        return DatasetListRequest(value.project_ref, value.directory_ref, value.cursor, value.limit)

    @staticmethod
    def _validate_request(value: DatasetValidateRequest) -> DatasetValidateRequest:
        if type(value) is not DatasetValidateRequest:
            raise TypeError("request must be exact DatasetValidateRequest")
        return DatasetValidateRequest(value.project_ref, value.dataset_ref, value.cursor, value.limit)

    @staticmethod
    def _observations_request(value: ObservationsRequest) -> ObservationsRequest:
        if type(value) is not ObservationsRequest:
            raise TypeError("request must be exact ObservationsRequest")
        stream = ObservationStreamRef.from_dict(value.stream.to_dict())
        if stream.family is not ObservationFamily.DATA:
            raise ValueError("observations request must name a data stream")
        return ObservationsRequest(stream, value.after_sequence, value.limit)

    # -- detach-and-revalidate discipline (RunsAPI._call, verbatim) --------------

    @staticmethod
    def _matches(current: object, baseline: object, rebuild) -> bool:
        try:
            return rebuild(current) == baseline
        except BaseException:
            return False

    @staticmethod
    def _changed() -> None:
        raise ValueError("data operation input changed during callback") from None

    @classmethod
    def _unchanged(cls, current: object, baseline: object, rebuild) -> None:
        if not cls._matches(current, baseline, rebuild):
            cls._changed()

    @classmethod
    def _call(cls, callback, original, baseline, presentation, rebuild, *extra):
        mutation_after_failure = False
        try:
            result = callback(presentation, *extra)
        except BaseException:
            mutation_after_failure = (
                not cls._matches(original, baseline, rebuild)
                or not cls._matches(presentation, baseline, rebuild)
            )
            if not mutation_after_failure:
                raise
        if mutation_after_failure:
            cls._changed()
        cls._unchanged(original, baseline, rebuild)
        cls._unchanged(presentation, baseline, rebuild)
        return result

    # -- result rebuilders ----------------------------------------------------------

    @staticmethod
    def _outcome(value: object, run: DataRunRef) -> DataOutcome:
        if type(value) is not DataOutcome:
            raise TypeError("data result must be exact DataOutcome")
        rebuilt = DataOutcome.from_dict(value.to_dict())
        if rebuilt.run != run:
            raise ValueError("data outcome does not bind the request")
        return rebuilt

    # -- verbs -----------------------------------------------------------------------

    def plan(self, request: DataRequest) -> DataPlan:
        baseline = self._request(request)
        presented = self._request(baseline)
        result = self._call(self._operations.plan, request, baseline, presented, self._request)
        if type(result) is not DataPlan:
            raise TypeError("data plan result must be exact DataPlan")
        rebuilt = DataPlan.from_dict(result.to_dict())
        if rebuilt.request != baseline:
            raise ValueError("data plan does not bind the request")
        self._unchanged(request, baseline, self._request)
        self._unchanged(presented, baseline, self._request)
        return rebuilt

    def preflight(self, plan: DataPlan) -> DataPreflight:
        baseline = self._plan(plan)
        presented = self._plan(baseline)
        result = self._call(self._operations.preflight, plan, baseline, presented, self._plan)
        if type(result) is not DataPreflight:
            raise TypeError("data preflight result must be exact DataPreflight")
        rebuilt = DataPreflight.from_dict(result.to_dict())
        if not rebuilt.binds(baseline):
            raise ValueError("data preflight does not bind the plan")
        self._unchanged(plan, baseline, self._plan)
        self._unchanged(presented, baseline, self._plan)
        return rebuilt

    def start(self, plan: DataPlan) -> DataStart:
        baseline = self._plan(plan)
        presented = self._plan(baseline)
        result = self._call(self._operations.start, plan, baseline, presented, self._plan)
        if type(result) is not DataStart:
            raise TypeError("data start result must be exact DataStart")
        rebuilt = DataStart.from_dict(result.to_dict())
        if rebuilt.run.project_ref != baseline.request.project_ref:
            raise ValueError("data start does not bind the plan")
        self._unchanged(plan, baseline, self._plan)
        self._unchanged(presented, baseline, self._plan)
        return rebuilt

    def show(self, run: DataRunRef) -> DataOutcome:
        baseline = self._run(run)
        presented = self._run(baseline)
        rebuilt = self._outcome(self._call(self._operations.show, run, baseline, presented, self._run), baseline)
        self._unchanged(run, baseline, self._run)
        self._unchanged(presented, baseline, self._run)
        return rebuilt

    def cancel(self, run: DataRunRef, reason: str) -> DataOutcome:
        baseline = self._run(run)
        presented = self._run(baseline)
        reason = _text(reason, "reason")
        rebuilt = self._outcome(self._call(self._operations.cancel, run, baseline, presented, self._run, reason), baseline)
        self._unchanged(run, baseline, self._run)
        self._unchanged(presented, baseline, self._run)
        return rebuilt

    def list(self, request: DataListRequest) -> DataPage:
        baseline = self._list_request(request)
        presented = self._list_request(baseline)
        result = self._call(self._operations.list, request, baseline, presented, self._list_request)
        if type(result) is not DataPage:
            raise TypeError("data list result must be exact DataPage")
        rebuilt = DataPage(
            self._list_request(result.request),
            tuple(DataOutcome.from_dict(item.to_dict()) for item in result.outcomes),
            result.next_cursor,
            result.truncated,
        )
        if rebuilt.request != baseline:
            raise ValueError("data list result does not bind the request")
        self._unchanged(request, baseline, self._list_request)
        self._unchanged(presented, baseline, self._list_request)
        return rebuilt

    def datasets(self, request: DatasetListRequest) -> DatasetPage:
        baseline = self._dataset_list_request(request)
        presented = self._dataset_list_request(baseline)
        result = self._call(self._operations.datasets, request, baseline, presented, self._dataset_list_request)
        if type(result) is not DatasetPage:
            raise TypeError("datasets result must be exact DatasetPage")
        rebuilt = DatasetPage(
            self._dataset_list_request(result.request),
            tuple(DatasetDescriptor.from_dict(item.to_dict()) for item in result.datasets),
            result.next_cursor,
            result.truncated,
        )
        if rebuilt.request != baseline:
            raise ValueError("datasets result does not bind the request")
        self._unchanged(request, baseline, self._dataset_list_request)
        self._unchanged(presented, baseline, self._dataset_list_request)
        return rebuilt

    def validate(self, request: DatasetValidateRequest) -> ValidationReport:
        baseline = self._validate_request(request)
        presented = self._validate_request(baseline)
        result = self._call(self._operations.validate, request, baseline, presented, self._validate_request)
        if type(result) is not ValidationReport:
            raise TypeError("validate result must be exact ValidationReport")
        rebuilt = ValidationReport(
            self._validate_request(result.request),
            result.rows_checked,
            result.rows_valid,
            tuple(ValidationFinding.from_dict(item.to_dict()) for item in result.findings),
            result.next_cursor,
            result.truncated,
        )
        if rebuilt.request != baseline:
            raise ValueError("validate result does not bind the request")
        self._unchanged(request, baseline, self._validate_request)
        self._unchanged(presented, baseline, self._validate_request)
        return rebuilt

    def observations(self, request: ObservationsRequest) -> ObservationPage:
        baseline = self._observations_request(request)
        presented = self._observations_request(baseline)
        result = self._call(self._operations.observations, request, baseline, presented, self._observations_request)
        if type(result) is not ObservationPage:
            raise TypeError("observation result must be exact ObservationPage")
        rebuilt = ObservationPage(
            self._observations_request(result.request),
            tuple(ObservationRecordV1.from_dict(item.to_dict()) for item in result.records),
            result.next_cursor,
            result.truncated,
        )
        if rebuilt.request != baseline:
            raise ValueError("observation result does not bind the request")
        self._unchanged(request, baseline, self._observations_request)
        self._unchanged(presented, baseline, self._observations_request)
        return rebuilt


__all__ = [
    "DataAPI", "DataListRequest", "DataMode", "DataOperationCode", "DataOperationError",
    "DataOperations", "DataOutcome", "DataPage", "DataPlan", "DataPreflight",
    "DataRequest", "DataResult", "DataRunRef", "DataRunState", "DataScenarioOutcome",
    "DataScenarioTarget", "DataStart", "DatasetDescriptor", "DatasetListRequest",
    "DatasetPage", "DatasetValidateRequest", "ValidationFinding", "ValidationFindingCode",
    "ValidationReport",
]
