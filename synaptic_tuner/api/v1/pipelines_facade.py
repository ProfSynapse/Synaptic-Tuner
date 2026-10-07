"""Pipeline planning, start, resume, cancel and observation facade (contract only).

Location: ``synaptic_tuner/api/v1/pipelines_facade.py``.

A pipeline is an ordered set of stages over one project. It **references**
child runs (a ``TrainingRunRef`` for ``train``, an ``EvaluationRunRef`` for
``evaluate``) and never owns them: every child keeps its own authoritative
state through ``RunsAPI`` or ``EvaluationAPI``, so a failed evaluation can
never rewrite a successful training as failed. Stage handoff is by
``VerifiedArtifact`` only, never by path: the outputs of one stage are the
inputs of the next, and each stage's ``attempt_key`` binds it to the exact
upstream artifact digests it consumed.

Closed vocabularies:

- ``PipelineStageName`` (five values) is the declared stage vocabulary; only
  ``train`` and ``evaluate`` are admitted (``ADMITTED_STAGES``). ``loss``,
  ``analysis`` and ``recommendation`` are declared but refused at plan time
  with ``stage_unsupported`` until their facades exist.
- ``PipelineState`` (eight values) is pipeline-only: ``partially_succeeded``
  is terminal with at least one succeeded and at least one failed or skipped
  stage; ``reconcile_required`` means at least one stage's child is in
  ``reconcile_required`` and no stage is running. No child run ever takes
  either state.
- ``StageState`` (seven values): ``planned`` (not started), ``running`` (child
  in flight), ``succeeded``, ``failed``, ``skipped`` (an upstream stage did not
  succeed so this stage never ran), ``cancelled`` and ``reconcile_required``
  (the child needs reconciliation before the stage can advance). The
  terminal members are ``succeeded``, ``failed``, ``skipped`` and
  ``cancelled``, matching ``PIPELINE_STAGE_OUTCOMES`` in ``observations.py``.
- ``PipelineOperationCode`` (ten values).

``attempt_key`` is ``contract_digest("synaptic-pipeline-stage-attempt/v1",
{pipeline_id, stage, spec_digest, input_digest})`` where ``input_digest`` is
``contract_digest("synaptic-pipeline-stage-inputs/v1", {"inputs": [...]})``
over the stage's input artifacts ordered by role (``stage_attempt_key``,
``stage_input_digest``). A ``PipelineRecord`` proves every stage's key
re-derives from its own inputs, so the record can never carry a stale key.

Every type here is a frozen slotted dataclass validating in ``__post_init__``
and round-tripping through ``exact_fields`` with canonical-JSON ``to_dict`` /
``from_dict``. ``PipelinesAPI`` applies the ``RunsAPI._call`` discipline
verbatim and refuses unadmitted stages before any callback runs.

Registered in the lazy export table of ``synaptic_tuner/api/v1/__init__.py``
and in both import-closure gates. Imports nothing from ``tuner.*``; the
reference implementation lives in ``api/v1/reference/pipelines.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from ._contract import contract_digest, digest_text, exact_fields, required_text
from .evaluation_facade import EvaluationRunRef
from .observations import (
    ObservationFamily,
    ObservationPage,
    ObservationRecordV1,
    ObservationStreamRef,
    ObservationsRequest,
)
from .providers import ProviderRef
from .results import TrainingRunRef, VerifiedArtifact


PIPELINE_RECORD_SCHEMA_VERSION = "synaptic-pipeline-record/v1"
PIPELINE_PLAN_SCHEMA_VERSION = "synaptic-pipeline-plan/v1"
PIPELINE_SPEC_DOMAIN = "synaptic-pipeline-spec/v1"
PIPELINE_STAGE_ATTEMPT_DOMAIN = "synaptic-pipeline-stage-attempt/v1"
PIPELINE_STAGE_INPUTS_DOMAIN = "synaptic-pipeline-stage-inputs/v1"

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


def _artifacts(value: object, name: str) -> tuple[VerifiedArtifact, ...]:
    if type(value) is not tuple or any(type(item) is not VerifiedArtifact for item in value):
        raise TypeError(f"{name} must be an exact tuple of VerifiedArtifact values")
    artifacts = tuple(VerifiedArtifact.from_dict(item.to_dict()) for item in value)
    roles = tuple(item.role for item in artifacts)
    if len(roles) != len(set(roles)):
        raise ValueError(f"{name} artifact roles must be unique")
    return tuple(sorted(artifacts, key=lambda item: item.role))


def _artifact_list(artifacts: tuple[VerifiedArtifact, ...]) -> list[dict[str, object]]:
    return [item.to_dict() for item in artifacts]


def _parse_artifact_list(value: object, name: str) -> tuple[VerifiedArtifact, ...]:
    return tuple(
        VerifiedArtifact.from_dict(item)  # type: ignore[arg-type]
        for item in _exact_list(value, name)
    )


def _parse_enum(enum_type: type, value: object, name: str):
    value = _text(value, name)
    try:
        return enum_type(value)
    except ValueError:
        raise ValueError(f"unknown {name}") from None


# --- identities and closed vocabularies ----------------------------------------


@dataclass(frozen=True, slots=True)
class PipelineRef:
    """Identity of one pipeline; never interchangeable with a run ref."""

    pipeline_id: str
    project_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "pipeline_id", _text(self.pipeline_id, "pipeline_id"))
        object.__setattr__(self, "project_ref", _text(self.project_ref, "project_ref"))

    def to_dict(self) -> dict[str, object]:
        return {"pipeline_id": self.pipeline_id, "project_ref": self.project_ref}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PipelineRef":
        value = exact_fields(value, frozenset({"pipeline_id", "project_ref"}), "pipeline_ref")
        return cls(value["pipeline_id"], value["project_ref"])  # type: ignore[arg-type]


class PipelineStageName(str, Enum):
    """The declared stage vocabulary; ``ADMITTED_STAGES`` names the admitted subset."""

    TRAIN = "train"
    EVALUATE = "evaluate"
    LOSS = "loss"
    ANALYSIS = "analysis"
    RECOMMENDATION = "recommendation"


ADMITTED_STAGES = frozenset({PipelineStageName.TRAIN, PipelineStageName.EVALUATE})


class PipelineState(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIALLY_SUCCEEDED = "partially_succeeded"
    FAILED = "failed"
    RECONCILE_REQUIRED = "reconcile_required"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"


class StageState(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    RECONCILE_REQUIRED = "reconcile_required"


TERMINAL_STAGE_STATES = frozenset({
    StageState.SUCCEEDED, StageState.FAILED, StageState.SKIPPED, StageState.CANCELLED,
})
TERMINAL_PIPELINE_STATES = frozenset({
    PipelineState.SUCCEEDED, PipelineState.PARTIALLY_SUCCEEDED, PipelineState.FAILED,
    PipelineState.CANCELLED,
})


class PipelineOperationCode(str, Enum):
    PIPELINE_MISSING = "pipeline_missing"
    CURSOR_INVALID = "cursor_invalid"
    SPEC_INVALID = "spec_invalid"
    STAGE_UNSUPPORTED = "stage_unsupported"
    STAGE_INPUT_MISSING = "stage_input_missing"
    STAGE_DIGEST_MISMATCH = "stage_digest_mismatch"
    RESUME_INELIGIBLE = "resume_ineligible"
    CANCEL_INELIGIBLE = "cancel_ineligible"
    STATE_CONFLICT = "state_conflict"
    INTEGRITY_ERROR = "integrity_error"


class PipelineOperationError(ValueError):
    def __init__(self, code: PipelineOperationCode) -> None:
        if type(code) is not PipelineOperationCode:
            raise TypeError("code must be exact PipelineOperationCode")
        self.code = code
        super().__init__(code.value)


# --- digests -----------------------------------------------------------------------


def stage_input_digest(inputs: tuple[VerifiedArtifact, ...]) -> str:
    """Digest of a stage's input artifacts, ordered by role; ``()`` has a digest too."""
    return contract_digest(
        PIPELINE_STAGE_INPUTS_DOMAIN, {"inputs": _artifact_list(_artifacts(inputs, "inputs"))}
    )


def stage_attempt_key(
    pipeline_id: str, stage: PipelineStageName, spec_digest: str, inputs: tuple[VerifiedArtifact, ...]
) -> str:
    """The resume key of one stage attempt: pipeline, stage, spec and exact inputs."""
    if type(stage) is not PipelineStageName:
        raise TypeError("stage must be exact PipelineStageName")
    return contract_digest(
        PIPELINE_STAGE_ATTEMPT_DOMAIN,
        {
            "pipeline_id": _text(pipeline_id, "pipeline_id"),
            "stage": stage.value,
            "spec_digest": digest_text(_text(spec_digest, "spec_digest"), "spec_digest"),
            "input_digest": stage_input_digest(inputs),
        },
    )


# --- stage specifications and the request ------------------------------------------


@dataclass(frozen=True, slots=True)
class PipelineTrainSpec:
    """What the ``train`` stage hands to ``TrainingAPI``: a canonical request and a provider."""

    canonical_json: str
    provider: ProviderRef

    def __post_init__(self) -> None:
        object.__setattr__(self, "canonical_json", _text(self.canonical_json, "canonical_json"))
        if type(self.provider) is not ProviderRef:
            raise TypeError("provider must be exact ProviderRef")
        object.__setattr__(self, "provider", ProviderRef.from_dict(self.provider.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {"canonical_json": self.canonical_json, "provider": self.provider.to_dict()}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PipelineTrainSpec":
        value = exact_fields(value, frozenset({"canonical_json", "provider"}), "pipeline_train_spec")
        provider = value["provider"]
        if type(provider) is not dict:
            raise TypeError("provider must be an exact object")
        return cls(value["canonical_json"], ProviderRef.from_dict(provider))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class PipelineEvaluateSpec:
    """What the ``evaluate`` stage hands to ``EvaluationAPI``.

    The evaluated model is bound to the upstream artifact, never named
    ``latest``: ``model_ref`` is the name the backend serves it under and
    ``model_revision`` is filled at run time with the sha256 of the input
    artifact whose role is ``model_artifact_role``.
    """

    backend: str
    scenario_refs: tuple[str, ...]
    model_ref: str
    model_artifact_role: str = "adapter"
    preset: str | None = None
    tags: tuple[str, ...] = ()
    case_limit: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", _text(self.backend, "backend"))
        object.__setattr__(self, "scenario_refs", _text_set(self.scenario_refs, "scenario_refs", minimum=1))
        object.__setattr__(self, "model_ref", _text(self.model_ref, "model_ref"))
        object.__setattr__(self, "model_artifact_role", _text(self.model_artifact_role, "model_artifact_role"))
        object.__setattr__(self, "preset", _optional_text(self.preset, "preset"))
        object.__setattr__(self, "tags", _text_set(self.tags, "tags"))
        if self.case_limit is not None:
            _count(self.case_limit, "case_limit", minimum=1)

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "scenario_refs": list(self.scenario_refs),
            "model_ref": self.model_ref,
            "model_artifact_role": self.model_artifact_role,
            "preset": self.preset,
            "tags": list(self.tags),
            "case_limit": self.case_limit,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PipelineEvaluateSpec":
        value = exact_fields(
            value,
            frozenset({
                "backend", "scenario_refs", "model_ref", "model_artifact_role", "preset", "tags", "case_limit",
            }),
            "pipeline_evaluate_spec",
        )
        return cls(
            value["backend"],  # type: ignore[arg-type]
            _exact_list(value["scenario_refs"], "scenario_refs"),  # type: ignore[arg-type]
            value["model_ref"],  # type: ignore[arg-type]
            value["model_artifact_role"],  # type: ignore[arg-type]
            value["preset"],  # type: ignore[arg-type]
            _exact_list(value["tags"], "tags"),  # type: ignore[arg-type]
            value["case_limit"],  # type: ignore[arg-type]
        )


_STAGE_SPEC_FIELDS = {PipelineStageName.TRAIN: "train", PipelineStageName.EVALUATE: "evaluate"}


@dataclass(frozen=True, slots=True)
class PipelineRequest:
    """An ordered stage set over one project with one spec per admitted stage.

    ``stages`` may name any declared stage so a request can be expressed
    honestly; admission is decided at plan time. A listed ``train`` or
    ``evaluate`` stage requires its spec and a spec requires its stage.
    ``evaluate`` must follow ``train`` because it consumes train outputs.
    """

    request_id: str
    project_ref: str
    stages: tuple[PipelineStageName, ...]
    train: PipelineTrainSpec | None = None
    evaluate: PipelineEvaluateSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _text(self.request_id, "request_id"))
        object.__setattr__(self, "project_ref", _text(self.project_ref, "project_ref"))
        if type(self.stages) is not tuple or any(type(item) is not PipelineStageName for item in self.stages):
            raise TypeError("stages must be an exact tuple of PipelineStageName values")
        if not self.stages:
            raise ValueError("stages requires at least 1 entry")
        if len(self.stages) != len(set(self.stages)):
            raise ValueError("stages must be unique")
        if PipelineStageName.EVALUATE in self.stages and (
            PipelineStageName.TRAIN not in self.stages
            or self.stages.index(PipelineStageName.TRAIN) > self.stages.index(PipelineStageName.EVALUATE)
        ):
            raise ValueError("evaluate must follow train")
        for stage, field in _STAGE_SPEC_FIELDS.items():
            spec = getattr(self, field)
            if (stage in self.stages) != (spec is not None):
                raise ValueError(f"{field} spec must be present exactly when the {stage.value} stage is listed")
        if self.train is not None:
            if type(self.train) is not PipelineTrainSpec:
                raise TypeError("train must be exact PipelineTrainSpec")
            object.__setattr__(self, "train", PipelineTrainSpec.from_dict(self.train.to_dict()))
        if self.evaluate is not None:
            if type(self.evaluate) is not PipelineEvaluateSpec:
                raise TypeError("evaluate must be exact PipelineEvaluateSpec")
            object.__setattr__(self, "evaluate", PipelineEvaluateSpec.from_dict(self.evaluate.to_dict()))

    @property
    def spec_digest(self) -> str:
        return contract_digest(PIPELINE_SPEC_DOMAIN, self.to_dict())

    @property
    def unadmitted_stages(self) -> tuple[PipelineStageName, ...]:
        return tuple(stage for stage in self.stages if stage not in ADMITTED_STAGES)

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "project_ref": self.project_ref,
            "stages": [stage.value for stage in self.stages],
            "train": None if self.train is None else self.train.to_dict(),
            "evaluate": None if self.evaluate is None else self.evaluate.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PipelineRequest":
        value = exact_fields(
            value, frozenset({"request_id", "project_ref", "stages", "train", "evaluate"}), "pipeline_request",
        )
        train, evaluate = value["train"], value["evaluate"]
        if train is not None and type(train) is not dict:
            raise TypeError("train must be an exact object or null")
        if evaluate is not None and type(evaluate) is not dict:
            raise TypeError("evaluate must be an exact object or null")
        return cls(
            value["request_id"],  # type: ignore[arg-type]
            value["project_ref"],  # type: ignore[arg-type]
            tuple(_parse_enum(PipelineStageName, item, "stage") for item in _exact_list(value["stages"], "stages")),
            None if train is None else PipelineTrainSpec.from_dict(train),
            None if evaluate is None else PipelineEvaluateSpec.from_dict(evaluate),
        )


@dataclass(frozen=True, slots=True)
class PipelinePlan:
    """A request an implementation has admitted; ``spec_digest`` is derived, never stored."""

    schema_version: str
    request: PipelineRequest

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != PIPELINE_PLAN_SCHEMA_VERSION:
            raise ValueError("unsupported pipeline plan schema version")
        if type(self.request) is not PipelineRequest:
            raise TypeError("request must be exact PipelineRequest")
        object.__setattr__(self, "request", PipelineRequest.from_dict(self.request.to_dict()))
        if self.request.unadmitted_stages:
            raise ValueError("a pipeline plan admits only train and evaluate stages")

    @property
    def spec_digest(self) -> str:
        return self.request.spec_digest

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "request": self.request.to_dict()}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PipelinePlan":
        value = exact_fields(value, frozenset({"schema_version", "request"}), "pipeline_plan")
        request = value["request"]
        if type(request) is not dict:
            raise TypeError("request must be an exact object")
        return cls(_text(value["schema_version"], "schema_version"), PipelineRequest.from_dict(request))


@dataclass(frozen=True, slots=True)
class PipelineStart:
    pipeline: PipelineRef
    accepted: bool

    def __post_init__(self) -> None:
        if type(self.pipeline) is not PipelineRef:
            raise TypeError("pipeline must be exact PipelineRef")
        object.__setattr__(self, "pipeline", PipelineRef.from_dict(self.pipeline.to_dict()))
        _bool(self.accepted, "accepted")

    def to_dict(self) -> dict[str, object]:
        return {"pipeline": self.pipeline.to_dict(), "accepted": self.accepted}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PipelineStart":
        value = exact_fields(value, frozenset({"pipeline", "accepted"}), "pipeline_start")
        pipeline = value["pipeline"]
        if type(pipeline) is not dict:
            raise TypeError("pipeline must be an exact object")
        return cls(PipelineRef.from_dict(pipeline), value["accepted"])  # type: ignore[arg-type]


# --- the record -------------------------------------------------------------------------

_RUN_REF_TYPES: dict[PipelineStageName, type | None] = {
    PipelineStageName.TRAIN: TrainingRunRef,
    PipelineStageName.EVALUATE: EvaluationRunRef,
    PipelineStageName.LOSS: None,
    PipelineStageName.ANALYSIS: None,
    PipelineStageName.RECOMMENDATION: None,
}


@dataclass(frozen=True, slots=True)
class PipelineStage:
    """One stage of a record: its state, resume key, child reference and artifact handoff.

    ``run`` is a ``TrainingRunRef`` for ``train`` and an ``EvaluationRunRef``
    for ``evaluate``; the stage name selects the type on parse. A ``planned``
    or ``skipped`` stage has no run and no outputs.
    """

    name: PipelineStageName
    state: StageState
    attempt_key: str
    run: TrainingRunRef | EvaluationRunRef | None = None
    inputs: tuple[VerifiedArtifact, ...] = ()
    outputs: tuple[VerifiedArtifact, ...] = ()

    def __post_init__(self) -> None:
        if type(self.name) is not PipelineStageName or type(self.state) is not StageState:
            raise TypeError("name/state have invalid types")
        object.__setattr__(self, "attempt_key", digest_text(_text(self.attempt_key, "attempt_key"), "attempt_key"))
        ref_type = _RUN_REF_TYPES[self.name]
        if self.run is not None:
            if ref_type is None or type(self.run) is not ref_type:
                raise TypeError(f"{self.name.value} stage run must be exact {getattr(ref_type, '__name__', 'None')}")
            object.__setattr__(self, "run", ref_type.from_dict(self.run.to_dict()))
        object.__setattr__(self, "inputs", _artifacts(self.inputs, "inputs"))
        object.__setattr__(self, "outputs", _artifacts(self.outputs, "outputs"))
        if self.state in (StageState.PLANNED, StageState.SKIPPED) and (self.run is not None or self.outputs):
            raise ValueError(f"a {self.state.value} stage has no run and no outputs")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name.value,
            "state": self.state.value,
            "attempt_key": self.attempt_key,
            "run": None if self.run is None else self.run.to_dict(),
            "inputs": _artifact_list(self.inputs),
            "outputs": _artifact_list(self.outputs),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PipelineStage":
        value = exact_fields(
            value, frozenset({"name", "state", "attempt_key", "run", "inputs", "outputs"}), "pipeline_stage",
        )
        name = _parse_enum(PipelineStageName, value["name"], "name")
        run = value["run"]
        if run is not None:
            if type(run) is not dict:
                raise TypeError("run must be an exact object or null")
            ref_type = _RUN_REF_TYPES[name]
            if ref_type is None:
                raise ValueError(f"{name.value} stage cannot reference a run")
            run = ref_type.from_dict(run)
        return cls(
            name,
            _parse_enum(StageState, value["state"], "state"),
            value["attempt_key"],  # type: ignore[arg-type]
            run,
            _parse_artifact_list(value["inputs"], "inputs"),
            _parse_artifact_list(value["outputs"], "outputs"),
        )


def _check_state(state: PipelineState, stages: tuple[PipelineStage, ...]) -> None:
    """The pipeline state must be consistent with its stages (§7.4 Q3)."""
    states = tuple(stage.state for stage in stages)
    all_terminal = all(item in TERMINAL_STAGE_STATES for item in states)
    succeeded = sum(1 for item in states if item is StageState.SUCCEEDED)
    failed_or_skipped = sum(1 for item in states if item in (StageState.FAILED, StageState.SKIPPED))
    cancelled = sum(1 for item in states if item is StageState.CANCELLED)
    if state is PipelineState.PLANNED:
        valid = all(item is StageState.PLANNED for item in states)
    elif state in (PipelineState.RUNNING, PipelineState.CANCEL_REQUESTED):
        valid = not all_terminal
    elif state is PipelineState.RECONCILE_REQUIRED:
        valid = StageState.RECONCILE_REQUIRED in states and StageState.RUNNING not in states
    elif state is PipelineState.SUCCEEDED:
        valid = succeeded == len(states)
    elif state is PipelineState.PARTIALLY_SUCCEEDED:
        valid = all_terminal and succeeded >= 1 and failed_or_skipped >= 1
    elif state is PipelineState.FAILED:
        valid = all_terminal and succeeded == 0 and failed_or_skipped >= 1
    else:
        valid = all_terminal and cancelled >= 1
    if not valid:
        raise ValueError(f"pipeline state {state.value} is inconsistent with its stages")


@dataclass(frozen=True, slots=True)
class PipelineRecord:
    """The public record of one pipeline: identity, spec, state, revision and stages.

    Every stage's ``attempt_key`` must re-derive from the pipeline id, its
    name, the spec digest and its own inputs, and the pipeline state must be
    consistent with the stage states. There is no diagnostic field: each
    child run carries its own.
    """

    schema_version: str
    pipeline: PipelineRef
    spec_digest: str
    state: PipelineState
    revision: int
    stages: tuple[PipelineStage, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != PIPELINE_RECORD_SCHEMA_VERSION:
            raise ValueError("unsupported pipeline record schema version")
        if type(self.pipeline) is not PipelineRef or type(self.state) is not PipelineState:
            raise TypeError("pipeline/state have invalid types")
        object.__setattr__(self, "pipeline", PipelineRef.from_dict(self.pipeline.to_dict()))
        object.__setattr__(self, "spec_digest", digest_text(_text(self.spec_digest, "spec_digest"), "spec_digest"))
        _count(self.revision, "revision", minimum=1)
        if type(self.stages) is not tuple or any(type(item) is not PipelineStage for item in self.stages):
            raise TypeError("stages must be an exact tuple of PipelineStage values")
        if not self.stages:
            raise ValueError("stages requires at least 1 entry")
        stages = tuple(PipelineStage.from_dict(item.to_dict()) for item in self.stages)
        names = tuple(stage.name for stage in stages)
        if len(names) != len(set(names)):
            raise ValueError("stage names must be unique")
        for stage in stages:
            expected = stage_attempt_key(self.pipeline.pipeline_id, stage.name, self.spec_digest, stage.inputs)
            if stage.attempt_key != expected:
                raise ValueError(f"{stage.name.value} attempt_key does not derive from its inputs")
            if stage.run is not None and stage.run.project_ref != self.pipeline.project_ref:
                raise ValueError(f"{stage.name.value} run project does not match the pipeline")
        _check_state(self.state, stages)
        object.__setattr__(self, "stages", stages)

    def stage(self, name: PipelineStageName) -> PipelineStage:
        for stage in self.stages:
            if stage.name is name:
                return stage
        raise KeyError(name.value)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "pipeline": self.pipeline.to_dict(),
            "spec_digest": self.spec_digest,
            "state": self.state.value,
            "revision": self.revision,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PipelineRecord":
        value = exact_fields(
            value,
            frozenset({"schema_version", "pipeline", "spec_digest", "state", "revision", "stages"}),
            "pipeline_record",
        )
        pipeline = value["pipeline"]
        if type(pipeline) is not dict:
            raise TypeError("pipeline must be an exact object")
        return cls(
            _text(value["schema_version"], "schema_version"),
            PipelineRef.from_dict(pipeline),
            value["spec_digest"],  # type: ignore[arg-type]
            _parse_enum(PipelineState, value["state"], "state"),
            value["revision"],  # type: ignore[arg-type]
            tuple(PipelineStage.from_dict(item) for item in _exact_list(value["stages"], "stages")),  # type: ignore[arg-type]
        )


# --- paging -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PipelineListRequest:
    project_ref: str
    cursor: str | None = None
    limit: int = _MAX_LIST_LIMIT

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_ref", _text(self.project_ref, "project_ref"))
        object.__setattr__(self, "cursor", _ascii_cursor(self.cursor))
        if type(self.limit) is not int or not 1 <= self.limit <= _MAX_LIST_LIMIT:
            raise ValueError(f"limit must be an integer from 1 through {_MAX_LIST_LIMIT}")


@dataclass(frozen=True, slots=True)
class PipelinePage:
    request: PipelineListRequest
    records: tuple[PipelineRecord, ...]
    next_cursor: str | None = None
    truncated: bool = False

    def __post_init__(self) -> None:
        if type(self.request) is not PipelineListRequest:
            raise TypeError("request must be exact PipelineListRequest")
        if type(self.records) is not tuple or any(type(item) is not PipelineRecord for item in self.records):
            raise TypeError("records must be an exact tuple of PipelineRecord")
        if len(self.records) > self.request.limit:
            raise ValueError("records exceed requested limit")
        if any(item.pipeline.project_ref != self.request.project_ref for item in self.records):
            raise ValueError("record project does not match list request")
        _bool(self.truncated, "truncated")
        object.__setattr__(self, "next_cursor", _ascii_cursor(self.next_cursor, "next_cursor"))
        if self.truncated != (self.next_cursor is not None):
            raise ValueError("next_cursor/truncated matrix invalid")
        if self.truncated and not self.records:
            raise ValueError("a truncated page must contain a record")


# --- operations and facade -------------------------------------------------------------


class PipelinesOperations(Protocol):
    def plan(self, request: PipelineRequest) -> PipelinePlan: ...
    def start(self, plan: PipelinePlan) -> PipelineStart: ...
    def show(self, pipeline: PipelineRef) -> PipelineRecord: ...
    def resume(self, pipeline: PipelineRef) -> PipelineRecord: ...
    def cancel(self, pipeline: PipelineRef, reason: str) -> PipelineRecord: ...
    def list(self, request: PipelineListRequest) -> PipelinePage: ...
    def observations(self, request: ObservationsRequest) -> ObservationPage: ...


class PipelinesAPI:
    """Public pipelines facade over host-composed ``PipelinesOperations``.

    Every verb rebuilds its input, presents a detached copy to the callback,
    re-validates the original and the copy after return or raise, and rebuilds
    the callback's result. ``plan`` refuses a request naming an unadmitted
    stage with ``stage_unsupported`` before any callback runs.
    """

    __slots__ = ("_operations",)

    def __init__(self, operations: PipelinesOperations) -> None:
        self._operations = operations

    # -- input rebuilders ------------------------------------------------------------

    @staticmethod
    def _pipeline(value: PipelineRef) -> PipelineRef:
        if type(value) is not PipelineRef:
            raise TypeError("pipeline must be exact PipelineRef")
        return PipelineRef.from_dict(value.to_dict())

    @staticmethod
    def _request(value: PipelineRequest) -> PipelineRequest:
        if type(value) is not PipelineRequest:
            raise TypeError("request must be exact PipelineRequest")
        return PipelineRequest.from_dict(value.to_dict())

    @staticmethod
    def _plan(value: PipelinePlan) -> PipelinePlan:
        if type(value) is not PipelinePlan:
            raise TypeError("plan must be exact PipelinePlan")
        return PipelinePlan.from_dict(value.to_dict())

    @staticmethod
    def _list_request(value: PipelineListRequest) -> PipelineListRequest:
        if type(value) is not PipelineListRequest:
            raise TypeError("request must be exact PipelineListRequest")
        return PipelineListRequest(value.project_ref, value.cursor, value.limit)

    @staticmethod
    def _observations_request(value: ObservationsRequest) -> ObservationsRequest:
        if type(value) is not ObservationsRequest:
            raise TypeError("request must be exact ObservationsRequest")
        stream = ObservationStreamRef.from_dict(value.stream.to_dict())
        if stream.family is not ObservationFamily.PIPELINE:
            raise ValueError("observations request must name a pipeline stream")
        return ObservationsRequest(stream, value.after_sequence, value.limit)

    # -- detach-and-revalidate discipline (RunsAPI._call, verbatim) ------------------

    @staticmethod
    def _matches(current: object, baseline: object, rebuild) -> bool:
        try:
            return rebuild(current) == baseline
        except BaseException:
            return False

    @staticmethod
    def _changed() -> None:
        raise ValueError("pipeline operation input changed during callback") from None

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

    # -- result rebuilders --------------------------------------------------------------

    @staticmethod
    def _record(value: object, pipeline: PipelineRef) -> PipelineRecord:
        if type(value) is not PipelineRecord:
            raise TypeError("pipeline result must be exact PipelineRecord")
        rebuilt = PipelineRecord.from_dict(value.to_dict())
        if rebuilt.pipeline != pipeline:
            raise ValueError("pipeline record does not bind the request")
        return rebuilt

    # -- verbs --------------------------------------------------------------------------

    def plan(self, request: PipelineRequest) -> PipelinePlan:
        baseline = self._request(request)
        if baseline.unadmitted_stages:
            raise PipelineOperationError(PipelineOperationCode.STAGE_UNSUPPORTED)
        presented = self._request(baseline)
        result = self._call(self._operations.plan, request, baseline, presented, self._request)
        if type(result) is not PipelinePlan:
            raise TypeError("pipeline plan result must be exact PipelinePlan")
        rebuilt = PipelinePlan.from_dict(result.to_dict())
        if rebuilt.request != baseline:
            raise ValueError("pipeline plan does not bind the request")
        self._unchanged(request, baseline, self._request)
        self._unchanged(presented, baseline, self._request)
        return rebuilt

    def start(self, plan: PipelinePlan) -> PipelineStart:
        baseline = self._plan(plan)
        presented = self._plan(baseline)
        result = self._call(self._operations.start, plan, baseline, presented, self._plan)
        if type(result) is not PipelineStart:
            raise TypeError("pipeline start result must be exact PipelineStart")
        rebuilt = PipelineStart.from_dict(result.to_dict())
        if rebuilt.pipeline.project_ref != baseline.request.project_ref:
            raise ValueError("pipeline start does not bind the plan")
        self._unchanged(plan, baseline, self._plan)
        self._unchanged(presented, baseline, self._plan)
        return rebuilt

    def show(self, pipeline: PipelineRef) -> PipelineRecord:
        baseline = self._pipeline(pipeline)
        presented = self._pipeline(baseline)
        rebuilt = self._record(self._call(self._operations.show, pipeline, baseline, presented, self._pipeline), baseline)
        self._unchanged(pipeline, baseline, self._pipeline)
        self._unchanged(presented, baseline, self._pipeline)
        return rebuilt

    def resume(self, pipeline: PipelineRef) -> PipelineRecord:
        baseline = self._pipeline(pipeline)
        presented = self._pipeline(baseline)
        rebuilt = self._record(self._call(self._operations.resume, pipeline, baseline, presented, self._pipeline), baseline)
        self._unchanged(pipeline, baseline, self._pipeline)
        self._unchanged(presented, baseline, self._pipeline)
        return rebuilt

    def cancel(self, pipeline: PipelineRef, reason: str) -> PipelineRecord:
        baseline = self._pipeline(pipeline)
        presented = self._pipeline(baseline)
        reason = _text(reason, "reason")
        rebuilt = self._record(
            self._call(self._operations.cancel, pipeline, baseline, presented, self._pipeline, reason), baseline,
        )
        self._unchanged(pipeline, baseline, self._pipeline)
        self._unchanged(presented, baseline, self._pipeline)
        return rebuilt

    def list(self, request: PipelineListRequest) -> PipelinePage:
        baseline = self._list_request(request)
        presented = self._list_request(baseline)
        result = self._call(self._operations.list, request, baseline, presented, self._list_request)
        if type(result) is not PipelinePage:
            raise TypeError("pipeline list result must be exact PipelinePage")
        rebuilt = PipelinePage(
            PipelineListRequest(result.request.project_ref, result.request.cursor, result.request.limit),
            tuple(PipelineRecord.from_dict(item.to_dict()) for item in result.records),
            result.next_cursor,
            result.truncated,
        )
        if rebuilt.request != baseline:
            raise ValueError("pipeline list result does not bind the request")
        self._unchanged(request, baseline, self._list_request)
        self._unchanged(presented, baseline, self._list_request)
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
    "ADMITTED_STAGES", "PipelineEvaluateSpec", "PipelineListRequest",
    "PipelineOperationCode", "PipelineOperationError", "PipelinePage", "PipelinePlan",
    "PipelineRecord", "PipelineRef", "PipelineRequest", "PipelineStage",
    "PipelineStageName", "PipelineStart", "PipelineState", "PipelineTrainSpec",
    "PipelinesAPI", "PipelinesOperations", "StageState", "stage_attempt_key",
    "stage_input_digest",
]
