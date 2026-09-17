"""Evaluation planning, start, observation and result facade (contract only).

Location: ``synaptic_tuner/api/v1/evaluation_facade.py``.

One model, one backend and one scenario set form one evaluation run with its
own ``EvaluationRunRef``. The ref is a distinct exact type, not a
``TrainingRunRef``, so an evaluation run can never reach ``RunsAPI`` and a
training run can never reach ``EvaluationAPI``; the store partition is the
discriminator, not a run-kind field.

Every type here is a frozen slotted dataclass validating in ``__post_init__``
and round-tripping through ``exact_fields`` with canonical-JSON ``to_dict`` /
``from_dict``. ``EvaluationAPI`` applies the ``RunsAPI._call`` discipline
verbatim: it rebuilds each input, hands the callback a detached copy, and
re-validates both the original and the copy after the callback returns or
raises. A result must bind the request it answers.

Closed vocabularies: ``EvaluationRunState`` (no ``reconcile_required``; an
indeterminate spend fails the run with ``spend_indeterminate``),
``JudgeVerdict`` (``inconclusive`` separates a broken judge from a bad model)
and ``EvaluationOperationCode``. Composite scores are ``ScoreV1`` values, not
ad hoc fields. The public result carries no response text: ``usage`` is
present only when measured, and artifacts are digest-verified references.

Registered in the lazy export table of ``synaptic_tuner/api/v1/__init__.py``
and in both import-closure gates. Imports nothing from ``tuner.*``; the
reference implementation over ``Evaluator/`` lives in ``api/v1/reference``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol

from ._contract import contract_digest, digest_text, exact_fields, exact_integer, required_text
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


EVALUATION_PLAN_SCHEMA_VERSION = "synaptic-evaluation-plan/v1"
EVALUATION_RESULT_SCHEMA_VERSION = "synaptic-evaluation-result/v1"

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


def _finite_float(value: object, name: str) -> float:
    if type(value) is bool or type(value) not in (int, float):
        raise TypeError(f"{name} must be a number")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return number


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


# --- identities and closed vocabularies ----------------------------------------


@dataclass(frozen=True, slots=True)
class EvaluationRunRef:
    """Identity of one evaluation run; never interchangeable with ``TrainingRunRef``."""

    run_id: str
    project_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", required_text(self.run_id, "run_id"))
        object.__setattr__(self, "project_ref", required_text(self.project_ref, "project_ref"))

    def to_dict(self) -> dict[str, object]:
        return {"run_id": self.run_id, "project_ref": self.project_ref}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "EvaluationRunRef":
        value = exact_fields(value, frozenset({"run_id", "project_ref"}), "evaluation_run_ref")
        return cls(run_id=value["run_id"], project_ref=value["project_ref"])  # type: ignore[arg-type]


class EvaluationRunState(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIALLY_SUCCEEDED = "partially_succeeded"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"


class JudgeVerdict(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class EvaluationOperationCode(str, Enum):
    RUN_MISSING = "run_missing"
    CURSOR_INVALID = "cursor_invalid"
    SCENARIO_INVALID = "scenario_invalid"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    BACKEND_UNMETERED = "backend_unmetered"
    JUDGE_FAILED = "judge_failed"
    SPEND_INDETERMINATE = "spend_indeterminate"
    CANCEL_INELIGIBLE = "cancel_ineligible"
    RESULT_UNAVAILABLE = "result_unavailable"
    STATE_CONFLICT = "state_conflict"
    INTEGRITY_ERROR = "integrity_error"


class EvaluationOperationError(ValueError):
    def __init__(self, code: EvaluationOperationCode) -> None:
        if type(code) is not EvaluationOperationCode:
            raise TypeError("code must be exact EvaluationOperationCode")
        self.code = code
        super().__init__(code.value)


@dataclass(frozen=True, slots=True)
class ScoreV1:
    """One named composite score, such as ``composite`` or ``quality_gated``."""

    name: str
    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "name"))
        object.__setattr__(self, "value", _finite_float(self.value, "value"))

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "value": self.value}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ScoreV1":
        value = exact_fields(value, frozenset({"name", "value"}), "score")
        return cls(value["name"], value["value"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class EvaluationModelRef:
    """The host-resolved, immutable model under evaluation."""

    model_ref: str
    model_revision: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_ref", _text(self.model_ref, "model_ref"))
        object.__setattr__(self, "model_revision", _text(self.model_revision, "model_revision"))

    def to_dict(self) -> dict[str, object]:
        return {"model_ref": self.model_ref, "model_revision": self.model_revision}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "EvaluationModelRef":
        value = exact_fields(value, frozenset({"model_ref", "model_revision"}), "model")
        return cls(value["model_ref"], value["model_revision"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class EvaluationVerdictCounts:
    """How many scored cases landed on each closed ``JudgeVerdict``."""

    passed: int
    failed: int
    inconclusive: int

    def __post_init__(self) -> None:
        for verdict in JudgeVerdict:
            object.__setattr__(self, verdict.value, _count(getattr(self, verdict.value), verdict.value))

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.inconclusive

    def to_dict(self) -> dict[str, object]:
        return {verdict.value: getattr(self, verdict.value) for verdict in JudgeVerdict}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "EvaluationVerdictCounts":
        value = exact_fields(value, frozenset(item.value for item in JudgeVerdict), "verdicts")
        return cls(value["passed"], value["failed"], value["inconclusive"])  # type: ignore[arg-type]


# --- request, plan, preflight, start -------------------------------------------


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    """One model, one backend, one scenario set.

    ``backend`` is an open name at the contract level; an implementation refuses
    an unknown one with ``backend_unavailable`` and an unmetered paid one with
    ``backend_unmetered``. Secrets never appear here: they are resolved by name
    through the host's ``SecretResolverPort`` at execution time.
    """

    request_id: str
    project_ref: str
    model: EvaluationModelRef
    backend: str
    scenario_refs: tuple[str, ...]
    preset: str | None = None
    tags: tuple[str, ...] = ()
    case_limit: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _text(self.request_id, "request_id"))
        object.__setattr__(self, "project_ref", _text(self.project_ref, "project_ref"))
        if type(self.model) is not EvaluationModelRef:
            raise TypeError("model must be exact EvaluationModelRef")
        object.__setattr__(self, "model", EvaluationModelRef.from_dict(self.model.to_dict()))
        object.__setattr__(self, "backend", _text(self.backend, "backend"))
        object.__setattr__(self, "scenario_refs", _text_set(self.scenario_refs, "scenario_refs", minimum=1))
        object.__setattr__(self, "preset", _optional_text(self.preset, "preset"))
        object.__setattr__(self, "tags", _text_set(self.tags, "tags"))
        if self.case_limit is not None:
            _count(self.case_limit, "case_limit", minimum=1)

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "project_ref": self.project_ref,
            "model": self.model.to_dict(),
            "backend": self.backend,
            "scenario_refs": list(self.scenario_refs),
            "preset": self.preset,
            "tags": list(self.tags),
            "case_limit": self.case_limit,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "EvaluationRequest":
        value = exact_fields(
            value,
            frozenset({
                "request_id", "project_ref", "model", "backend", "scenario_refs",
                "preset", "tags", "case_limit",
            }),
            "evaluation_request",
        )
        return cls(
            value["request_id"],  # type: ignore[arg-type]
            value["project_ref"],  # type: ignore[arg-type]
            EvaluationModelRef.from_dict(value["model"]),  # type: ignore[arg-type]
            value["backend"],  # type: ignore[arg-type]
            _exact_list(value["scenario_refs"], "scenario_refs"),  # type: ignore[arg-type]
            value["preset"],  # type: ignore[arg-type]
            _exact_list(value["tags"], "tags"),  # type: ignore[arg-type]
            value["case_limit"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    """A resolved request: how many cases it yields and the digest of their sources.

    ``plan_fingerprint`` is derived, never stored, so it always reflects the
    plan's current field values; ``EvaluationPreflight`` binds to it.
    """

    schema_version: str
    request: EvaluationRequest
    cases_total: int
    scenario_digest: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != EVALUATION_PLAN_SCHEMA_VERSION:
            raise ValueError("unsupported evaluation plan schema version")
        if type(self.request) is not EvaluationRequest:
            raise TypeError("request must be exact EvaluationRequest")
        object.__setattr__(self, "request", EvaluationRequest.from_dict(self.request.to_dict()))
        _count(self.cases_total, "cases_total", minimum=1)
        object.__setattr__(self, "scenario_digest", digest_text(_text(self.scenario_digest, "scenario_digest"), "scenario_digest"))

    @property
    def plan_fingerprint(self) -> str:
        return contract_digest(EVALUATION_PLAN_SCHEMA_VERSION, self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "request": self.request.to_dict(),
            "cases_total": self.cases_total,
            "scenario_digest": self.scenario_digest,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "EvaluationPlan":
        value = exact_fields(
            value,
            frozenset({"schema_version", "request", "cases_total", "scenario_digest"}),
            "evaluation_plan",
        )
        return cls(
            _text(value["schema_version"], "schema_version"),
            EvaluationRequest.from_dict(value["request"]),  # type: ignore[arg-type]
            value["cases_total"],  # type: ignore[arg-type]
            value["scenario_digest"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class EvaluationPreflight:
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

    def binds(self, plan: EvaluationPlan) -> bool:
        if type(plan) is not EvaluationPlan:
            raise TypeError("plan must be exact EvaluationPlan")
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
    def from_dict(cls, value: dict[str, object]) -> "EvaluationPreflight":
        value = exact_fields(
            value,
            frozenset({
                "plan_fingerprint", "ready", "checked_at", "expires_at",
                "authorization", "diagnostic_codes",
            }),
            "evaluation_preflight",
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
class EvaluationStart:
    run: EvaluationRunRef
    accepted: bool

    def __post_init__(self) -> None:
        if type(self.run) is not EvaluationRunRef:
            raise TypeError("run must be exact EvaluationRunRef")
        object.__setattr__(self, "run", EvaluationRunRef.from_dict(self.run.to_dict()))
        _bool(self.accepted, "accepted")

    def to_dict(self) -> dict[str, object]:
        return {"run": self.run.to_dict(), "accepted": self.accepted}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "EvaluationStart":
        value = exact_fields(value, frozenset({"run", "accepted"}), "evaluation_start")
        return cls(EvaluationRunRef.from_dict(value["run"]), value["accepted"])  # type: ignore[arg-type]


# --- outcome, paging, result ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    """The durable state view of one run; ``result`` carries the scores."""

    run: EvaluationRunRef
    state: EvaluationRunState
    cases_total: int
    cases_scored: int
    artifacts: tuple[VerifiedArtifact, ...] = ()
    diagnostic_code: str | None = None

    def __post_init__(self) -> None:
        if type(self.run) is not EvaluationRunRef or type(self.state) is not EvaluationRunState:
            raise TypeError("run/state have invalid types")
        object.__setattr__(self, "run", EvaluationRunRef.from_dict(self.run.to_dict()))
        _count(self.cases_total, "cases_total")
        _count(self.cases_scored, "cases_scored")
        if self.cases_scored > self.cases_total:
            raise ValueError("cases_scored must not exceed cases_total")
        object.__setattr__(self, "artifacts", _artifacts(self.artifacts))
        object.__setattr__(self, "diagnostic_code", _optional_text(self.diagnostic_code, "diagnostic_code"))

    def to_dict(self) -> dict[str, object]:
        return {
            "run": self.run.to_dict(),
            "state": self.state.value,
            "cases_total": self.cases_total,
            "cases_scored": self.cases_scored,
            "artifacts": _artifact_list(self.artifacts),
            "diagnostic_code": self.diagnostic_code,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "EvaluationOutcome":
        value = exact_fields(
            value,
            frozenset({"run", "state", "cases_total", "cases_scored", "artifacts", "diagnostic_code"}),
            "evaluation_outcome",
        )
        return cls(
            EvaluationRunRef.from_dict(value["run"]),  # type: ignore[arg-type]
            _parse_enum(EvaluationRunState, value["state"], "state"),
            value["cases_total"],  # type: ignore[arg-type]
            value["cases_scored"],  # type: ignore[arg-type]
            _parse_artifact_list(value["artifacts"]),
            value["diagnostic_code"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class EvaluationListRequest:
    project_ref: str
    cursor: str | None = None
    limit: int = _MAX_LIST_LIMIT

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_ref", _text(self.project_ref, "project_ref"))
        object.__setattr__(self, "cursor", _ascii_cursor(self.cursor))
        if type(self.limit) is not int or not 1 <= self.limit <= _MAX_LIST_LIMIT:
            raise ValueError(f"limit must be an integer from 1 through {_MAX_LIST_LIMIT}")


@dataclass(frozen=True, slots=True)
class EvaluationPage:
    request: EvaluationListRequest
    outcomes: tuple[EvaluationOutcome, ...]
    next_cursor: str | None = None
    truncated: bool = False

    def __post_init__(self) -> None:
        if type(self.request) is not EvaluationListRequest:
            raise TypeError("request must be exact EvaluationListRequest")
        if type(self.outcomes) is not tuple or any(type(item) is not EvaluationOutcome for item in self.outcomes):
            raise TypeError("outcomes must be an exact tuple of EvaluationOutcome")
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


@dataclass(frozen=True, slots=True)
class EvaluationResultRequest:
    run: EvaluationRunRef

    def __post_init__(self) -> None:
        if type(self.run) is not EvaluationRunRef:
            raise TypeError("run must be exact EvaluationRunRef")
        object.__setattr__(self, "run", EvaluationRunRef.from_dict(self.run.to_dict()))


_RESULT_FIELDS = frozenset({
    "schema_version", "run", "state", "backend", "model", "cases_total", "cases_scored",
    "scores", "verdicts", "artifacts", "diagnostic_code",
})


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """The public record of one run: counts, composites, verdicts, verified artifacts.

    Response text never enters this record. ``usage`` is present only when the
    backend measured it; an ``unavailable`` usage is refused here and the field
    is omitted from the canonical document rather than published as null.
    """

    schema_version: str
    run: EvaluationRunRef
    state: EvaluationRunState
    backend: str
    model: EvaluationModelRef
    cases_total: int
    cases_scored: int
    scores: tuple[ScoreV1, ...]
    verdicts: EvaluationVerdictCounts
    artifacts: tuple[VerifiedArtifact, ...] = ()
    diagnostic_code: str | None = None
    usage: UsageRecordV1 | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != EVALUATION_RESULT_SCHEMA_VERSION:
            raise ValueError("unsupported evaluation result schema version")
        if type(self.run) is not EvaluationRunRef or type(self.state) is not EvaluationRunState:
            raise TypeError("run/state have invalid types")
        object.__setattr__(self, "run", EvaluationRunRef.from_dict(self.run.to_dict()))
        object.__setattr__(self, "backend", _text(self.backend, "backend"))
        if type(self.model) is not EvaluationModelRef:
            raise TypeError("model must be exact EvaluationModelRef")
        object.__setattr__(self, "model", EvaluationModelRef.from_dict(self.model.to_dict()))
        _count(self.cases_total, "cases_total")
        _count(self.cases_scored, "cases_scored")
        if self.cases_scored > self.cases_total:
            raise ValueError("cases_scored must not exceed cases_total")
        if type(self.scores) is not tuple or any(type(item) is not ScoreV1 for item in self.scores):
            raise TypeError("scores must be an exact tuple of ScoreV1 values")
        scores = tuple(ScoreV1.from_dict(item.to_dict()) for item in self.scores)
        names = tuple(item.name for item in scores)
        if len(names) != len(set(names)):
            raise ValueError("score names must be unique")
        object.__setattr__(self, "scores", tuple(sorted(scores, key=lambda item: item.name)))
        if type(self.verdicts) is not EvaluationVerdictCounts:
            raise TypeError("verdicts must be exact EvaluationVerdictCounts")
        object.__setattr__(self, "verdicts", EvaluationVerdictCounts.from_dict(self.verdicts.to_dict()))
        if self.verdicts.total != self.cases_scored:
            raise ValueError("verdict counts must sum to cases_scored")
        object.__setattr__(self, "artifacts", _artifacts(self.artifacts))
        object.__setattr__(self, "diagnostic_code", _optional_text(self.diagnostic_code, "diagnostic_code"))
        if self.usage is not None:
            if type(self.usage) is not UsageRecordV1:
                raise TypeError("usage must be exact UsageRecordV1 or None")
            if self.usage.availability is not UsageAvailability.MEASURED:
                raise ValueError("unavailable usage must be omitted, not published")
            object.__setattr__(self, "usage", UsageRecordV1.from_dict(self.usage.to_dict()))

    def to_dict(self) -> dict[str, object]:
        document: dict[str, object] = {
            "schema_version": self.schema_version,
            "run": self.run.to_dict(),
            "state": self.state.value,
            "backend": self.backend,
            "model": self.model.to_dict(),
            "cases_total": self.cases_total,
            "cases_scored": self.cases_scored,
            "scores": [item.to_dict() for item in self.scores],
            "verdicts": self.verdicts.to_dict(),
            "artifacts": _artifact_list(self.artifacts),
            "diagnostic_code": self.diagnostic_code,
        }
        if self.usage is not None:
            document["usage"] = self.usage.to_dict()
        return document

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "EvaluationResult":
        if type(value) is not dict:
            raise TypeError("evaluation_result must be an exact object")
        keys = tuple(dict.keys(value))
        if any(type(key) is not str for key in keys):
            raise TypeError("evaluation_result field names must be exact strings")
        expected = _RESULT_FIELDS | {"usage"} if "usage" in keys else _RESULT_FIELDS
        value = exact_fields(value, expected, "evaluation_result")
        usage = value.get("usage")
        if "usage" in value and usage is None:
            raise ValueError("usage must be omitted when unavailable, not null")
        return cls(
            _text(value["schema_version"], "schema_version"),
            EvaluationRunRef.from_dict(value["run"]),  # type: ignore[arg-type]
            _parse_enum(EvaluationRunState, value["state"], "state"),
            value["backend"],  # type: ignore[arg-type]
            EvaluationModelRef.from_dict(value["model"]),  # type: ignore[arg-type]
            value["cases_total"],  # type: ignore[arg-type]
            value["cases_scored"],  # type: ignore[arg-type]
            tuple(ScoreV1.from_dict(item) for item in _exact_list(value["scores"], "scores")),  # type: ignore[arg-type]
            EvaluationVerdictCounts.from_dict(value["verdicts"]),  # type: ignore[arg-type]
            _parse_artifact_list(value["artifacts"]),
            value["diagnostic_code"],  # type: ignore[arg-type]
            None if usage is None else UsageRecordV1.from_dict(usage),  # type: ignore[arg-type]
        )


# --- operations and facade -------------------------------------------------------


class EvaluationOperations(Protocol):
    def plan(self, request: EvaluationRequest) -> EvaluationPlan: ...
    def preflight(self, plan: EvaluationPlan) -> EvaluationPreflight: ...
    def start(self, plan: EvaluationPlan) -> EvaluationStart: ...
    def show(self, run: EvaluationRunRef) -> EvaluationOutcome: ...
    def cancel(self, run: EvaluationRunRef, reason: str) -> EvaluationOutcome: ...
    def list(self, request: EvaluationListRequest) -> EvaluationPage: ...
    def result(self, request: EvaluationResultRequest) -> EvaluationResult: ...
    def observations(self, request: ObservationsRequest) -> ObservationPage: ...


class EvaluationAPI:
    """Public evaluation facade over host-composed ``EvaluationOperations``.

    Every verb rebuilds its input, presents a detached copy to the callback,
    re-validates the original and the copy after return or raise, and rebuilds
    the callback's result so nothing the callback retains can alias the values
    handed back to the caller.
    """

    __slots__ = ("_operations",)

    def __init__(self, operations: EvaluationOperations) -> None:
        self._operations = operations

    # -- input rebuilders (one per accepted input type) --------------------------

    @staticmethod
    def _run(value: EvaluationRunRef) -> EvaluationRunRef:
        if type(value) is not EvaluationRunRef:
            raise TypeError("run must be exact EvaluationRunRef")
        return EvaluationRunRef.from_dict(value.to_dict())

    @staticmethod
    def _request(value: EvaluationRequest) -> EvaluationRequest:
        if type(value) is not EvaluationRequest:
            raise TypeError("request must be exact EvaluationRequest")
        return EvaluationRequest.from_dict(value.to_dict())

    @staticmethod
    def _plan(value: EvaluationPlan) -> EvaluationPlan:
        if type(value) is not EvaluationPlan:
            raise TypeError("plan must be exact EvaluationPlan")
        return EvaluationPlan.from_dict(value.to_dict())

    @staticmethod
    def _list_request(value: EvaluationListRequest) -> EvaluationListRequest:
        if type(value) is not EvaluationListRequest:
            raise TypeError("request must be exact EvaluationListRequest")
        return EvaluationListRequest(value.project_ref, value.cursor, value.limit)

    @classmethod
    def _result_request(cls, value: EvaluationResultRequest) -> EvaluationResultRequest:
        if type(value) is not EvaluationResultRequest:
            raise TypeError("request must be exact EvaluationResultRequest")
        return EvaluationResultRequest(cls._run(value.run))

    @staticmethod
    def _observations_request(value: ObservationsRequest) -> ObservationsRequest:
        if type(value) is not ObservationsRequest:
            raise TypeError("request must be exact ObservationsRequest")
        stream = ObservationStreamRef.from_dict(value.stream.to_dict())
        if stream.family is not ObservationFamily.EVALUATION:
            raise ValueError("observations request must name an evaluation stream")
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
        raise ValueError("evaluation operation input changed during callback") from None

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
    def _outcome(value: object, run: EvaluationRunRef) -> EvaluationOutcome:
        if type(value) is not EvaluationOutcome:
            raise TypeError("evaluation result must be exact EvaluationOutcome")
        rebuilt = EvaluationOutcome.from_dict(value.to_dict())
        if rebuilt.run != run:
            raise ValueError("evaluation outcome does not bind the request")
        return rebuilt

    # -- verbs -----------------------------------------------------------------------

    def plan(self, request: EvaluationRequest) -> EvaluationPlan:
        baseline = self._request(request)
        presented = self._request(baseline)
        result = self._call(self._operations.plan, request, baseline, presented, self._request)
        if type(result) is not EvaluationPlan:
            raise TypeError("evaluation plan result must be exact EvaluationPlan")
        rebuilt = EvaluationPlan.from_dict(result.to_dict())
        if rebuilt.request != baseline:
            raise ValueError("evaluation plan does not bind the request")
        self._unchanged(request, baseline, self._request)
        self._unchanged(presented, baseline, self._request)
        return rebuilt

    def preflight(self, plan: EvaluationPlan) -> EvaluationPreflight:
        baseline = self._plan(plan)
        presented = self._plan(baseline)
        result = self._call(self._operations.preflight, plan, baseline, presented, self._plan)
        if type(result) is not EvaluationPreflight:
            raise TypeError("evaluation preflight result must be exact EvaluationPreflight")
        rebuilt = EvaluationPreflight.from_dict(result.to_dict())
        if not rebuilt.binds(baseline):
            raise ValueError("evaluation preflight does not bind the plan")
        self._unchanged(plan, baseline, self._plan)
        self._unchanged(presented, baseline, self._plan)
        return rebuilt

    def start(self, plan: EvaluationPlan) -> EvaluationStart:
        baseline = self._plan(plan)
        presented = self._plan(baseline)
        result = self._call(self._operations.start, plan, baseline, presented, self._plan)
        if type(result) is not EvaluationStart:
            raise TypeError("evaluation start result must be exact EvaluationStart")
        rebuilt = EvaluationStart.from_dict(result.to_dict())
        if rebuilt.run.project_ref != baseline.request.project_ref:
            raise ValueError("evaluation start does not bind the plan")
        self._unchanged(plan, baseline, self._plan)
        self._unchanged(presented, baseline, self._plan)
        return rebuilt

    def show(self, run: EvaluationRunRef) -> EvaluationOutcome:
        baseline = self._run(run)
        presented = self._run(baseline)
        rebuilt = self._outcome(self._call(self._operations.show, run, baseline, presented, self._run), baseline)
        self._unchanged(run, baseline, self._run)
        self._unchanged(presented, baseline, self._run)
        return rebuilt

    def cancel(self, run: EvaluationRunRef, reason: str) -> EvaluationOutcome:
        baseline = self._run(run)
        presented = self._run(baseline)
        reason = _text(reason, "reason")
        rebuilt = self._outcome(self._call(self._operations.cancel, run, baseline, presented, self._run, reason), baseline)
        self._unchanged(run, baseline, self._run)
        self._unchanged(presented, baseline, self._run)
        return rebuilt

    def list(self, request: EvaluationListRequest) -> EvaluationPage:
        baseline = self._list_request(request)
        presented = self._list_request(baseline)
        result = self._call(self._operations.list, request, baseline, presented, self._list_request)
        if type(result) is not EvaluationPage:
            raise TypeError("evaluation list result must be exact EvaluationPage")
        rebuilt = EvaluationPage(
            EvaluationListRequest(result.request.project_ref, result.request.cursor, result.request.limit),
            tuple(EvaluationOutcome.from_dict(item.to_dict()) for item in result.outcomes),
            result.next_cursor,
            result.truncated,
        )
        if rebuilt.request != baseline:
            raise ValueError("evaluation list result does not bind the request")
        self._unchanged(request, baseline, self._list_request)
        self._unchanged(presented, baseline, self._list_request)
        return rebuilt

    def result(self, request: EvaluationResultRequest) -> EvaluationResult:
        baseline = self._result_request(request)
        presented = self._result_request(baseline)
        result = self._call(self._operations.result, request, baseline, presented, self._result_request)
        if type(result) is not EvaluationResult:
            raise TypeError("evaluation result must be exact EvaluationResult")
        rebuilt = EvaluationResult.from_dict(result.to_dict())
        if rebuilt.run != baseline.run:
            raise ValueError("evaluation result does not bind the request")
        self._unchanged(request, baseline, self._result_request)
        self._unchanged(presented, baseline, self._result_request)
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
    "EvaluationAPI", "EvaluationListRequest", "EvaluationModelRef",
    "EvaluationOperationCode", "EvaluationOperationError", "EvaluationOperations",
    "EvaluationOutcome", "EvaluationPage", "EvaluationPlan", "EvaluationPreflight",
    "EvaluationRequest", "EvaluationResult", "EvaluationResultRequest",
    "EvaluationRunRef", "EvaluationRunState", "EvaluationStart",
    "EvaluationVerdictCounts", "JudgeVerdict", "ScoreV1",
]
