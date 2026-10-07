"""Pipelines reference composition: ``PipelinesOperations`` over the child families.

Location: ``synaptic_tuner/api/v1/reference/pipelines.py``.

``PipelinesOperations`` (``api/v1/pipelines_facade.py``: plan, start, show,
resume, cancel, list, observations) is implemented here by
``ReferencePipelineOperationsV1``. A pipeline **references** child runs and
never owns them: the ``train`` stage is driven through the host's
``TrainingOperations`` and ``RunsOperations`` (wrapped in the public
``TrainingAPI`` / ``RunsAPI`` so every child result is rebuilt and bound), the
``evaluate`` stage through ``EvaluationOperations`` (``EvaluationAPI``). Stage
handoff is by ``VerifiedArtifact`` only: the train child's verified artifacts
become the evaluate stage's inputs and the evaluated model's ``model_revision``
is the sha256 of the artifact named by the spec's ``model_artifact_role``. No
path is ever constructed here; no legacy tracking directory and no project
state directory is read or written.

Durable layout (partition -> key -> document):

- ``pipeline`` / ``plan/<project_digest>/<pipeline_key>``: the admitted
  ``PipelinePlan`` (``put_if_absent``), so ``resume`` on a cold host can
  rebuild every stage request from the spec the digest names.
- ``pipeline`` / ``pipeline/<project_digest>/<pipeline_key>``: the head
  document on the record port, holding the current ``PipelineRecord``, its
  digest and the sequence and rolling chain digest of the lifecycle event that
  produced it. The record's ``revision`` equals the store revision.
- ``pipeline`` / same key on the stream port: the lifecycle events
  (``created``, ``started``, ``stage_started``, ``stage_blocked``,
  ``stage_finished``, ``resumed``, ``cancel_requested``). At most ``pipeline_event_budget(n)``
  (8 plus 6 per stage) durable events are ever appended; the writer asserts
  the budget before every append.
- ``observation`` / ``pipeline/<project_digest>/<pipeline_key>``: the public
  ``pipeline_stage_started`` / ``pipeline_stage_completed`` records (§4.2).

Driving. ``start`` creates the record and drives it synchronously: the train
child is polled through ``RunsAPI.outcome`` (at most
``maximum_child_polls`` polls per call, with the host's optional pacing hook
between polls), verified, and its artifacts recorded as outputs; the evaluate
child runs to completion inside ``EvaluationAPI.start``. A pipeline whose poll
budget ran out, or whose process died, stays ``running`` with the child
referenced; ``resume`` re-attaches to that child without starting another.
``resume`` recomputes every ``attempt_key``: a succeeded stage whose key
matches is kept and never re-driven; a stage whose inputs changed (its
upstream child's live artifacts differ from what was recorded) is re-planned
with the new inputs and a new key; a failed evaluate stage is retried with a
fresh child; and when an upstream child's live artifacts no longer match the
inputs a *succeeded* downstream stage was built on, ``resume`` refuses with
``stage_digest_mismatch`` rather than rewriting a success. A failed or cancelled train stage is not retried:
training run identity belongs to the training family, so such a pipeline is
``resume_ineligible``. ``cancel`` requests cancellation on the in-flight
child only and marks stages that never started ``cancelled``; completed
children are never touched.

An evaluation that ends ``partially_succeeded`` completes its stage as
``succeeded``: the evaluation ran and produced its result, and the verdict
detail lives in the child. A train child needing reconciliation puts its stage
and the pipeline in ``reconcile_required``; the operator reconciles the child
through ``RunsAPI`` and then resumes the pipeline.

Consumed by ``synaptic_tuner/api/v1/reference/composition.py`` and exported
lazily through ``synaptic_tuner/api/v1/reference/__init__.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from threading import RLock
from typing import Protocol

from synaptic_tuner.api.v1.evaluation_facade import (
    EvaluationAPI,
    EvaluationModelRef,
    EvaluationRequest,
    EvaluationRunRef,
    EvaluationRunState,
)
from synaptic_tuner.api.v1.observations import (
    OBSERVATION_SCHEMA_VERSION,
    ObservationFamily,
    ObservationKind,
    ObservationPage,
    ObservationRecordV1,
    ObservationsRequest,
    ObservationStreamRef,
    PipelineStageCompletedPayloadV1,
    PipelineStageStartedPayloadV1,
)
from synaptic_tuner.api.v1.pipelines_facade import (
    PIPELINE_PLAN_SCHEMA_VERSION,
    PIPELINE_RECORD_SCHEMA_VERSION,
    TERMINAL_PIPELINE_STATES,
    TERMINAL_STAGE_STATES,
    PipelineListRequest,
    PipelineOperationCode,
    PipelineOperationError,
    PipelinePage,
    PipelinePlan,
    PipelineRecord,
    PipelineRef,
    PipelineRequest,
    PipelineStage,
    PipelineStageName,
    PipelineStart,
    PipelineState,
    StageState,
    stage_attempt_key,
)
from synaptic_tuner.api.v1.ports import (
    DurableRecordStorePort,
    DurableStreamStorePort,
    StoragePartition,
    StoredRecordV1,
    StoredStreamPageV1,
)
from synaptic_tuner.api.v1.results import TrainingRunRef, TrainingRunState, VerifiedArtifact
from synaptic_tuner.api.v1.runs_facade import RunsAPI
from synaptic_tuner.api.v1.training_facade import TrainingAPI
from tuner.execution.foundation_v2.canonical import digest_text, domain_digest

from .provider_family import require_methods


# --- constants ---------------------------------------------------------------------

PIPELINE_HEAD_SCHEMA = "synaptic-reference-pipeline-head/v1"
PIPELINE_EVENT_SCHEMA = "synaptic-reference-pipeline-event/v1"
PIPELINE_CHAIN_DOMAIN = "synaptic-reference-pipeline-chain/v1"
PIPELINE_ID_DOMAIN = "synaptic-reference-pipeline-id/v1"
PIPELINE_KEY_DOMAIN = "synaptic-reference-pipeline-key/v1"
PIPELINE_PROJECT_DOMAIN = "synaptic-reference-pipeline-project-key/v1"
PIPELINE_EVENT_BUDGET_BASE = 8
PIPELINE_EVENT_BUDGET_PER_STAGE = 6
PIPELINE_HEAD_MAXIMUM_BYTES = 64 * 1024
DEFAULT_MAXIMUM_CHILD_POLLS = 10_000
HISTORY_PAGE_LIMIT = 200
APPEND_ATTEMPTS = 4

_PIPELINE = StoragePartition.PIPELINE.value
_OBSERVATION = StoragePartition.OBSERVATION.value
_EVENT_KINDS = frozenset({
    "created", "started", "stage_started", "stage_blocked", "stage_finished", "resumed", "cancel_requested",
})
_HEAD_KEYS = frozenset({"schema_version", "revision", "sequence", "chain_digest", "record_digest", "record"})
_ENTRY_KEYS = frozenset({
    "schema_version", "revision", "previous_sequence", "previous_chain_digest",
    "event", "record_digest", "chain_digest",
})
_EVENT_KEYS = frozenset({"kind", "occurred_at", "state"})
_IN_FLIGHT = frozenset({StageState.RUNNING, StageState.RECONCILE_REQUIRED})
_RESUME_INELIGIBLE = frozenset({PipelineState.PLANNED, PipelineState.SUCCEEDED, PipelineState.CANCELLED})
_TRAIN_IN_FLIGHT = frozenset({
    TrainingRunState.PLANNED, TrainingRunState.QUEUED, TrainingRunState.RUNNING, TrainingRunState.CANCEL_REQUESTED,
})
_EVALUATION_IN_FLIGHT = frozenset({
    EvaluationRunState.PLANNED, EvaluationRunState.RUNNING, EvaluationRunState.CANCEL_REQUESTED,
})


def pipeline_event_budget(stage_count: int) -> int:
    """Durable events one pipeline may ever append: 8 plus 6 per stage (§4.3)."""
    return PIPELINE_EVENT_BUDGET_BASE + PIPELINE_EVENT_BUDGET_PER_STAGE * stage_count


def _error(code: PipelineOperationCode) -> PipelineOperationError:
    return PipelineOperationError(code)


def _integrity() -> PipelineOperationError:
    return _error(PipelineOperationCode.INTEGRITY_ERROR)


# --- host ports ---------------------------------------------------------------------


class PipelinePacingPort(Protocol):
    """Called between two polls of an in-flight child; a host sleeps or backs off here."""

    def wait(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ReferencePipelinePortsV1:
    """What a host injects to compose the pipelines family.

    ``maximum_child_polls`` bounds how many times one ``start`` or ``resume``
    call polls an in-flight child before returning with the pipeline still
    ``running``; ``pacing`` runs between polls.
    """

    maximum_child_polls: int = DEFAULT_MAXIMUM_CHILD_POLLS
    pacing: PipelinePacingPort | None = None

    def __post_init__(self) -> None:
        if type(self.maximum_child_polls) is not int or self.maximum_child_polls < 1:
            raise ValueError("maximum_child_polls must be a positive integer")
        if self.pacing is not None:
            require_methods(self.pacing, "wait")


# --- canonical documents ------------------------------------------------------------


def _dump(document: dict[str, object]) -> bytes:
    try:
        encoded = json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise _integrity() from None
    if len(encoded) > PIPELINE_HEAD_MAXIMUM_BYTES:
        raise _integrity()
    return encoded


def _load(raw: bytes, schema_version: str) -> dict[str, object]:
    if type(raw) is not bytes or not raw or len(raw) > PIPELINE_HEAD_MAXIMUM_BYTES:
        raise _integrity()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise _integrity() from None
    if type(document) is not dict or document.get("schema_version") != schema_version:
        raise _integrity()
    if _dump(document) != raw:
        raise _integrity()
    return document


def _project_digest(project_ref: str) -> str:
    return domain_digest(PIPELINE_PROJECT_DOMAIN, project_ref.encode("utf-8"))


def _pipeline_key(pipeline_id: str) -> str:
    return domain_digest(PIPELINE_KEY_DOMAIN, pipeline_id.encode("utf-8"))


def _pipeline_id(plan: PipelinePlan) -> str:
    return "pl-" + domain_digest(PIPELINE_ID_DOMAIN, plan.spec_digest.encode("ascii"))[:32]


def _head_key(pipeline: PipelineRef) -> str:
    return f"pipeline/{_project_digest(pipeline.project_ref)}/{_pipeline_key(pipeline.pipeline_id)}"


def _plan_key(pipeline: PipelineRef) -> str:
    return f"plan/{_project_digest(pipeline.project_ref)}/{_pipeline_key(pipeline.pipeline_id)}"


def _observation_key(pipeline: PipelineRef) -> str:
    return f"pipeline/{_project_digest(pipeline.project_ref)}/{_pipeline_key(pipeline.pipeline_id)}"


def _stream_ref(pipeline: PipelineRef) -> ObservationStreamRef:
    return ObservationStreamRef(ObservationFamily.PIPELINE, pipeline.project_ref, pipeline.pipeline_id)


def _record_digest(record: PipelineRecord) -> str:
    return domain_digest(PIPELINE_RECORD_SCHEMA_VERSION, _dump(record.to_dict()))


def _event_digest(event: dict[str, object]) -> str:
    return domain_digest(PIPELINE_EVENT_SCHEMA, _dump(event))


def _chain_digest(previous: str | None, event_digest: str, record_digest: str) -> str:
    return domain_digest(
        PIPELINE_CHAIN_DOMAIN,
        _dump({"previous": previous, "event": event_digest, "record": record_digest}),
    )


@dataclass(frozen=True, slots=True)
class _Head:
    """A validated head: the record plus the store revision and the chain position."""

    record: PipelineRecord
    revision: int
    sequence: int
    chain_digest: str
    raw: bytes


# --- record algebra -------------------------------------------------------------------


def _planned_stage(pipeline: PipelineRef, spec_digest: str, name: PipelineStageName) -> PipelineStage:
    return PipelineStage(name, StageState.PLANNED, stage_attempt_key(pipeline.pipeline_id, name, spec_digest, ()))


def _derive_state(stages: tuple[PipelineStage, ...], *, cancel_requested: bool) -> PipelineState:
    """The pipeline state implied by its stages; ``cancel_requested`` keeps the cancel intent."""
    states = tuple(stage.state for stage in stages)
    if all(state is StageState.PLANNED for state in states):
        return PipelineState.PLANNED
    if not all(state in TERMINAL_STAGE_STATES for state in states):
        if cancel_requested:
            return PipelineState.CANCEL_REQUESTED
        if StageState.RUNNING not in states and StageState.RECONCILE_REQUIRED in states:
            return PipelineState.RECONCILE_REQUIRED
        return PipelineState.RUNNING
    if StageState.CANCELLED in states:
        return PipelineState.CANCELLED
    if all(state is StageState.SUCCEEDED for state in states):
        return PipelineState.SUCCEEDED
    if StageState.SUCCEEDED in states:
        return PipelineState.PARTIALLY_SUCCEEDED
    return PipelineState.FAILED


def _with_stage(record: PipelineRecord, replacement: PipelineStage, *, revision: int, cancel_requested: bool) -> PipelineRecord:
    stages = tuple(replacement if stage.name is replacement.name else stage for stage in record.stages)
    return _rebuild(record, stages, revision=revision, cancel_requested=cancel_requested)


def _rebuild(record: PipelineRecord, stages: tuple[PipelineStage, ...], *, revision: int, cancel_requested: bool) -> PipelineRecord:
    return PipelineRecord(
        PIPELINE_RECORD_SCHEMA_VERSION, record.pipeline, record.spec_digest,
        _derive_state(stages, cancel_requested=cancel_requested), revision, stages,
    )


def _resolve_later_stages(
    record: PipelineRecord, from_index: int, state: StageState
) -> tuple[PipelineStage, ...]:
    """Every planned stage after ``from_index`` becomes ``state`` (skipped or cancelled)."""
    stages = []
    for index, stage in enumerate(record.stages):
        if index > from_index and stage.state is StageState.PLANNED:
            stage = PipelineStage(stage.name, state, stage.attempt_key, None, stage.inputs, ())
        stages.append(stage)
    return tuple(stages)


# --- stream helpers -------------------------------------------------------------------


def _read_page(
    streams: DurableStreamStorePort, partition: str, key: str, after_sequence: int | None, limit: int
) -> StoredStreamPageV1:
    try:
        page = streams.read_page(partition=partition, stream_key=key, after_sequence=after_sequence, limit=limit)
    except Exception:
        raise _integrity() from None
    if type(page) is not StoredStreamPageV1:
        raise _integrity()
    return page


def _tail_sequence(streams: DurableStreamStorePort, partition: str, key: str) -> int | None:
    tail = None
    cursor = None
    while True:
        page = _read_page(streams, partition, key, cursor, HISTORY_PAGE_LIMIT)
        if page.entries:
            tail = page.entries[-1].sequence
        if not page.truncated or page.next_cursor is None:
            return tail
        cursor = page.next_cursor


class _ObservationWriter:
    """Appends ``pipeline_*`` observation records for one pipeline to the stream port."""

    __slots__ = ("_streams", "_key", "_stream", "_clock", "_next")

    def __init__(self, streams: DurableStreamStorePort, pipeline: PipelineRef, clock) -> None:
        self._streams = streams
        self._key = _observation_key(pipeline)
        self._stream = _stream_ref(pipeline)
        self._clock = clock
        self._next = (_tail_sequence(streams, _OBSERVATION, self._key) or 0) + 1

    def emit(self, kind: ObservationKind, payload: object) -> None:
        for _ in range(APPEND_ATTEMPTS):
            record = ObservationRecordV1(
                OBSERVATION_SCHEMA_VERSION, self._stream, self._next, self._clock.now(), kind, payload
            )
            canonical = _dump(record.to_dict())
            try:
                appended = self._streams.append(
                    partition=_OBSERVATION, stream_key=self._key, sequence=self._next, canonical=canonical
                )
            except Exception:
                raise _integrity() from None
            if appended is True:
                self._next += 1
                return
            if appended is not False:
                raise _integrity()
            self._next = (_tail_sequence(self._streams, _OBSERVATION, self._key) or 0) + 1
        raise _integrity()

    def started(self, stage: PipelineStage) -> None:
        self.emit(ObservationKind.PIPELINE_STAGE_STARTED, PipelineStageStartedPayloadV1(stage.name.value, stage.attempt_key))

    def completed(self, stage: PipelineStage) -> None:
        if stage.state in TERMINAL_STAGE_STATES:
            self.emit(
                ObservationKind.PIPELINE_STAGE_COMPLETED,
                PipelineStageCompletedPayloadV1(stage.name.value, stage.attempt_key, stage.state.value),
            )


# --- operations ---------------------------------------------------------------------


class ReferencePipelineOperationsV1:
    """``PipelinesOperations`` over the durable stores and the child family operations."""

    __slots__ = (
        "_records", "_streams", "_clock", "_training", "_runs", "_evaluation", "_ports",
        "_validated", "_lock", "_driving",
    )

    def __init__(
        self,
        *,
        records: DurableRecordStorePort,
        streams: DurableStreamStorePort,
        clock,
        training,
        runs,
        evaluation,
        ports: ReferencePipelinePortsV1,
    ) -> None:
        require_methods(records, "create", "read", "compare_and_swap", "put_if_absent", "list_page")
        require_methods(streams, "append", "read_page")
        require_methods(clock, "now")
        require_methods(training, "load", "resolve", "plan", "preflight", "start")
        require_methods(runs, "show", "outcome", "cancel", "verify")
        if evaluation is not None:
            require_methods(evaluation, "plan", "start", "show", "cancel")
        if type(ports) is not ReferencePipelinePortsV1:
            raise TypeError("exact ReferencePipelinePortsV1 required")
        self._records = records
        self._streams = streams
        self._clock = clock
        self._training = TrainingAPI(training, clock=clock)
        self._runs = RunsAPI(runs)
        self._evaluation = None if evaluation is None else EvaluationAPI(evaluation)
        self._ports = ports
        self._validated: dict[str, _Head] = {}
        self._lock = RLock()
        self._driving: set[str] = set()

    # -- head documents ------------------------------------------------------------

    def _read_stored(self, key: str) -> StoredRecordV1 | None:
        try:
            stored = self._records.read(partition=_PIPELINE, key=key)
        except Exception:
            raise _integrity() from None
        if stored is None:
            return None
        if type(stored) is not StoredRecordV1 or stored.key != key:
            raise _integrity()
        return stored

    def _entries(self, key: str, until_sequence: int) -> list[dict[str, object]]:
        found: list[dict[str, object]] = []
        cursor = None
        while True:
            page = _read_page(self._streams, _PIPELINE, key, cursor, HISTORY_PAGE_LIMIT)
            for entry in page.entries:
                if entry.sequence > until_sequence:
                    return found
                found.append(self._decode_entry(entry.sequence, entry.canonical))
            if not page.truncated or page.next_cursor is None or page.next_cursor >= until_sequence:
                return found
            cursor = page.next_cursor

    @staticmethod
    def _decode_entry(sequence: int, raw: bytes) -> dict[str, object]:
        document = _load(raw, PIPELINE_EVENT_SCHEMA)
        if set(document) != _ENTRY_KEYS:
            raise _integrity()
        event = document["event"]
        if type(event) is not dict or set(event) != _EVENT_KEYS or event["kind"] not in _EVENT_KINDS:
            raise _integrity()
        if type(document["revision"]) is not int or document["revision"] < 1:
            raise _integrity()
        try:
            PipelineState(event["state"])
            digest_text(document["record_digest"], "record_digest")
            digest_text(document["chain_digest"], "chain_digest")
            if document["previous_chain_digest"] is not None:
                digest_text(document["previous_chain_digest"], "previous_chain_digest")
        except Exception:
            raise _integrity() from None
        document["sequence"] = sequence
        return document

    def _validate(self, key: str, stored: StoredRecordV1) -> _Head:
        """Decode the head, replay the event chain and prove the head sits on it."""
        with self._lock:
            cached = self._validated.get(key)
            if cached is not None and cached.raw == stored.canonical and cached.revision == stored.revision:
                return cached
        document = _load(stored.canonical, PIPELINE_HEAD_SCHEMA)
        if set(document) != _HEAD_KEYS:
            raise _integrity()
        revision, sequence, chain = document["revision"], document["sequence"], document["chain_digest"]
        if type(revision) is not int or type(sequence) is not int or revision < 1 or sequence < 1:
            raise _integrity()
        if revision != stored.revision or type(chain) is not str:
            raise _integrity()
        try:
            record = PipelineRecord.from_dict(document["record"])  # type: ignore[arg-type]
        except Exception:
            raise _integrity() from None
        if record.revision != revision or _record_digest(record) != document["record_digest"]:
            raise _integrity()
        if _head_key(record.pipeline) != key:
            raise _integrity()
        entries = self._entries(key, sequence)
        if len(entries) != sequence or len(entries) > pipeline_event_budget(len(record.stages)):
            raise _integrity()
        previous_chain: str | None = None
        previous_sequence: int | None = None
        for expected, entry in enumerate(entries, start=1):
            if entry["sequence"] != expected:
                raise _integrity()
            if entry["previous_sequence"] != previous_sequence or entry["previous_chain_digest"] != previous_chain:
                raise _integrity()
            event_digest = _event_digest(entry["event"])  # type: ignore[arg-type]
            computed = _chain_digest(previous_chain, event_digest, entry["record_digest"])  # type: ignore[arg-type]
            if computed != entry["chain_digest"]:
                raise _integrity()
            previous_chain, previous_sequence = computed, expected
        last = entries[-1]
        if previous_chain != chain or last["event"]["state"] != record.state.value:  # type: ignore[index]
            raise _integrity()
        if last["revision"] != revision or last["record_digest"] != document["record_digest"]:
            raise _integrity()
        head = _Head(record, revision, sequence, chain, stored.canonical)
        with self._lock:
            self._validated[key] = head
        return head

    def _head(self, pipeline: PipelineRef) -> _Head | None:
        key = _head_key(pipeline)
        stored = self._read_stored(key)
        return None if stored is None else self._validate(key, stored)

    def _require_head(self, pipeline: PipelineRef) -> _Head:
        head = self._head(pipeline)
        if head is None:
            raise _error(PipelineOperationCode.PIPELINE_MISSING)
        return head

    @staticmethod
    def _head_document(record: PipelineRecord, *, sequence: int, chain: str) -> bytes:
        return _dump(
            {
                "schema_version": PIPELINE_HEAD_SCHEMA,
                "revision": record.revision,
                "sequence": sequence,
                "chain_digest": chain,
                "record_digest": _record_digest(record),
                "record": record.to_dict(),
            }
        )

    def _append_event(self, key: str, *, kind: str, record: PipelineRecord, previous: _Head | None) -> tuple[int, str]:
        """Append one lifecycle event; asserts the per-pipeline durable event budget."""
        previous_sequence = None if previous is None else previous.sequence
        previous_chain = None if previous is None else previous.chain_digest
        sequence = (previous_sequence or 0) + 1
        if sequence > pipeline_event_budget(len(record.stages)):
            raise _integrity()
        event = {"kind": kind, "occurred_at": self._clock.now(), "state": record.state.value}
        record_digest = _record_digest(record)
        chain = _chain_digest(previous_chain, _event_digest(event), record_digest)
        canonical = _dump(
            {
                "schema_version": PIPELINE_EVENT_SCHEMA,
                "revision": record.revision,
                "previous_sequence": previous_sequence,
                "previous_chain_digest": previous_chain,
                "event": event,
                "record_digest": record_digest,
                "chain_digest": chain,
            }
        )
        try:
            appended = self._streams.append(partition=_PIPELINE, stream_key=key, sequence=sequence, canonical=canonical)
        except Exception:
            raise _integrity() from None
        if appended is True:
            return sequence, chain
        if appended is False:
            raise _error(PipelineOperationCode.STATE_CONFLICT)
        raise _integrity()

    def _create(self, record: PipelineRecord) -> _Head | None:
        """Genesis: event 1 then the head at revision 1; ``None`` when another writer won."""
        if record.revision != 1:
            raise _integrity()
        key = _head_key(record.pipeline)
        sequence, chain = self._append_event(key, kind="created", record=record, previous=None)
        raw = self._head_document(record, sequence=sequence, chain=chain)
        try:
            created = self._records.create(partition=_PIPELINE, key=key, canonical=raw)
        except Exception:
            raise _integrity() from None
        if created is not True:
            return None
        head = _Head(record, 1, sequence, chain, raw)
        with self._lock:
            self._validated[key] = head
        return head

    def _transition(self, head: _Head, record: PipelineRecord, kind: str) -> _Head:
        """Append the event, then commit through the head compare-and-swap."""
        if record.revision != head.revision + 1:
            raise _integrity()
        key = _head_key(record.pipeline)
        sequence, chain = self._append_event(key, kind=kind, record=record, previous=head)
        raw = self._head_document(record, sequence=sequence, chain=chain)
        try:
            swapped = self._records.compare_and_swap(
                partition=_PIPELINE, key=key, expected_revision=head.revision, canonical=raw
            )
        except Exception:
            raise _integrity() from None
        if swapped is not True:
            raise _error(PipelineOperationCode.STATE_CONFLICT)
        updated = _Head(record, record.revision, sequence, chain, raw)
        with self._lock:
            self._validated[key] = updated
        return updated

    # -- stored plan ------------------------------------------------------------------

    def _store_plan(self, pipeline: PipelineRef, plan: PipelinePlan) -> None:
        canonical = _dump(plan.to_dict())
        try:
            admitted = self._records.put_if_absent(partition=_PIPELINE, key=_plan_key(pipeline), canonical=canonical)
        except Exception:
            raise _integrity() from None
        if admitted is not True:
            raise _error(PipelineOperationCode.STATE_CONFLICT)

    def _plan_for(self, head: _Head) -> PipelinePlan:
        stored = self._read_stored(_plan_key(head.record.pipeline))
        if stored is None:
            raise _integrity()
        try:
            plan = PipelinePlan.from_dict(_load(stored.canonical, PIPELINE_PLAN_SCHEMA_VERSION))
        except PipelineOperationError:
            raise
        except Exception:
            raise _integrity() from None
        if plan.spec_digest != head.record.spec_digest or _pipeline_id(plan) != head.record.pipeline.pipeline_id:
            raise _integrity()
        return plan

    # -- child families ---------------------------------------------------------------

    def _evaluation_api(self) -> EvaluationAPI:
        if self._evaluation is None:
            raise _error(PipelineOperationCode.STAGE_UNSUPPORTED)
        return self._evaluation

    @staticmethod
    def _evaluation_request(pipeline: PipelineRef, request: PipelineRequest, stage: PipelineStage, revision: int) -> EvaluationRequest:
        spec = request.evaluate
        if spec is None:
            raise _error(PipelineOperationCode.STAGE_INPUT_MISSING)
        model_artifact = tuple(item for item in stage.inputs if item.role == spec.model_artifact_role)
        if len(model_artifact) != 1:
            raise _error(PipelineOperationCode.STAGE_INPUT_MISSING)
        return EvaluationRequest(
            f"{pipeline.pipeline_id}/evaluate/{stage.attempt_key[:16]}/{revision}",
            pipeline.project_ref,
            EvaluationModelRef(spec.model_ref, model_artifact[0].sha256),
            spec.backend,
            spec.scenario_refs,
            spec.preset,
            spec.tags,
            spec.case_limit,
        )

    def _start_train_child(self, request: PipelineRequest) -> TrainingRunRef | None:
        """Run the training ladder; ``None`` when the child could not be started."""
        spec = request.train
        if spec is None:
            return None
        try:
            loaded = self._training.load(spec.canonical_json)
            if loaded.project_ref != request.project_ref:
                return None
            plan = self._training.plan(self._training.resolve(loaded), spec.provider)
            started = self._training.start(plan, self._training.preflight(plan))
        except Exception:
            return None
        if started.accepted is not True or started.run.project_ref != request.project_ref:
            return None
        return started.run

    def _train_stage_outcome(self, stage: PipelineStage, *, poll: bool) -> PipelineStage:
        """Project the train child's state onto the stage; ``poll`` observes the provider."""
        run = stage.run
        assert type(run) is TrainingRunRef
        try:
            outcome = self._runs.outcome(run) if poll else self._runs.show(run)
        except Exception:
            return self._finish(stage, StageState.FAILED)
        state = outcome.state
        if state in _TRAIN_IN_FLIGHT:
            return PipelineStage(stage.name, StageState.RUNNING, stage.attempt_key, run, stage.inputs, ())
        if state is TrainingRunState.RECONCILE_REQUIRED:
            return PipelineStage(stage.name, StageState.RECONCILE_REQUIRED, stage.attempt_key, run, stage.inputs, ())
        if state is TrainingRunState.CANCELLED:
            return self._finish(stage, StageState.CANCELLED)
        if state is TrainingRunState.FAILED:
            return self._finish(stage, StageState.FAILED)
        try:
            verified = self._runs.verify(run).verified if not outcome.artifacts else True
            artifacts = outcome.artifacts if outcome.artifacts else self._runs.show(run).artifacts
        except Exception:
            return self._finish(stage, StageState.FAILED)
        if verified is not True or not artifacts:
            return self._finish(stage, StageState.FAILED)
        return self._finish(stage, StageState.SUCCEEDED, artifacts)

    def _evaluate_stage_outcome(self, stage: PipelineStage, run: EvaluationRunRef) -> PipelineStage:
        try:
            outcome = self._evaluation_api().show(run)
        except Exception:
            return self._finish(stage, StageState.FAILED, run=run)
        state = outcome.state
        if state in _EVALUATION_IN_FLIGHT:
            return PipelineStage(stage.name, StageState.RUNNING, stage.attempt_key, run, stage.inputs, ())
        if state is EvaluationRunState.CANCELLED:
            return self._finish(stage, StageState.CANCELLED, run=run)
        if state is EvaluationRunState.FAILED:
            return self._finish(stage, StageState.FAILED, run=run)
        return self._finish(stage, StageState.SUCCEEDED, outcome.artifacts, run=run)

    @staticmethod
    def _finish(
        stage: PipelineStage, state: StageState, outputs: tuple[VerifiedArtifact, ...] = (), *, run=None
    ) -> PipelineStage:
        return PipelineStage(stage.name, state, stage.attempt_key, run if run is not None else stage.run, stage.inputs, outputs)

    def _live_outputs(self, stage: PipelineStage) -> tuple[VerifiedArtifact, ...]:
        """The child's current artifacts; the record's outputs must still match them."""
        try:
            if type(stage.run) is TrainingRunRef:
                return self._runs.show(stage.run).artifacts
            if type(stage.run) is EvaluationRunRef:
                return self._evaluation_api().show(stage.run).artifacts
        except PipelineOperationError:
            raise
        except Exception:
            raise _integrity() from None
        raise _integrity()

    # -- the driver -------------------------------------------------------------------

    def _drive(self, pipeline: PipelineRef, plan: PipelinePlan) -> None:
        with self._lock:
            if pipeline.pipeline_id in self._driving:
                raise _error(PipelineOperationCode.STATE_CONFLICT)
            self._driving.add(pipeline.pipeline_id)
        try:
            self._advance(pipeline, plan.request)
        finally:
            with self._lock:
                self._driving.discard(pipeline.pipeline_id)

    def _advance(self, pipeline: PipelineRef, request: PipelineRequest) -> None:
        """Advance the first non-terminal stage until the pipeline settles or the poll budget ends."""
        observations = _ObservationWriter(self._streams, pipeline, self._clock)
        polls = 0
        while True:
            head = self._require_head(pipeline)
            record = head.record
            if record.state in TERMINAL_PIPELINE_STATES:
                return
            cancel_requested = record.state is PipelineState.CANCEL_REQUESTED
            index = next((i for i, stage in enumerate(record.stages) if stage.state not in TERMINAL_STAGE_STATES), None)
            if index is None:
                raise _integrity()
            stage = record.stages[index]
            revision = head.revision + 1

            if stage.state is StageState.PLANNED:
                if cancel_requested:
                    stages = _resolve_later_stages(record, index - 1, StageState.CANCELLED)
                    updated = _rebuild(record, stages, revision=revision, cancel_requested=True)
                    self._transition(head, updated, "stage_finished")
                    for item in updated.stages[index:]:
                        observations.completed(item)
                    continue
                upstream = record.stages[index - 1] if index > 0 else None
                if upstream is not None and upstream.state is not StageState.SUCCEEDED:
                    stages = _resolve_later_stages(record, index - 1, StageState.SKIPPED)
                    updated = _rebuild(record, stages, revision=revision, cancel_requested=False)
                    self._transition(head, updated, "stage_finished")
                    for item in updated.stages[index:]:
                        observations.completed(item)
                    continue
                inputs = upstream.outputs if upstream is not None and stage.name is PipelineStageName.EVALUATE else ()
                key = stage_attempt_key(pipeline.pipeline_id, stage.name, record.spec_digest, inputs)
                if stage.name is PipelineStageName.TRAIN:
                    run = self._start_train_child(request)
                    if run is None:
                        failed = PipelineStage(stage.name, StageState.FAILED, key, None, inputs, ())
                        updated = _with_stage(record, failed, revision=revision, cancel_requested=False)
                        updated = _rebuild(updated, _resolve_later_stages(updated, index, StageState.SKIPPED), revision=revision, cancel_requested=False)
                        self._transition(head, updated, "stage_finished")
                        for item in updated.stages[index:]:
                            observations.completed(item)
                        continue
                    running = PipelineStage(stage.name, StageState.RUNNING, key, run, inputs, ())
                    head = self._transition(head, _with_stage(record, running, revision=revision, cancel_requested=False), "stage_started")
                    observations.started(running)
                    continue
                running = PipelineStage(stage.name, StageState.RUNNING, key, None, inputs, ())
                head = self._transition(head, _with_stage(record, running, revision=revision, cancel_requested=False), "stage_started")
                observations.started(running)
                continue

            if stage.name is PipelineStageName.TRAIN:
                if stage.state is StageState.RUNNING:
                    if polls >= self._ports.maximum_child_polls:
                        return
                    if polls and self._ports.pacing is not None:
                        self._ports.pacing.wait()
                    polls += 1
                    projected = self._train_stage_outcome(stage, poll=True)
                else:
                    projected = self._train_stage_outcome(stage, poll=False)
            else:
                if stage.run is None and cancel_requested:
                    projected = self._finish(stage, StageState.CANCELLED)
                elif stage.run is None:
                    projected = self._run_evaluate_child(pipeline, request, stage, head.revision)
                else:
                    projected = self._evaluate_stage_outcome(stage, stage.run)  # type: ignore[arg-type]

            if projected == stage:
                if stage.state is StageState.RECONCILE_REQUIRED:
                    return
                continue
            updated = _with_stage(record, projected, revision=revision, cancel_requested=cancel_requested)
            if projected.state in TERMINAL_STAGE_STATES and projected.state is not StageState.SUCCEEDED:
                later = StageState.CANCELLED if cancel_requested else StageState.SKIPPED
                updated = _rebuild(updated, _resolve_later_stages(updated, index, later), revision=revision, cancel_requested=cancel_requested)
            if projected.state in TERMINAL_STAGE_STATES:
                kind = "stage_finished"
            elif projected.state is StageState.RECONCILE_REQUIRED:
                kind = "stage_blocked"
            else:
                kind = "stage_started"
            self._transition(head, updated, kind)
            if projected.state in TERMINAL_STAGE_STATES:
                for item in updated.stages[index:]:
                    observations.completed(item)
            if projected.state is StageState.RECONCILE_REQUIRED:
                return

    def _run_evaluate_child(
        self, pipeline: PipelineRef, request: PipelineRequest, stage: PipelineStage, revision: int
    ) -> PipelineStage:
        """Plan and start the evaluate child for a running stage whose reference is not yet known."""
        try:
            api = self._evaluation_api()
            evaluation_request = self._evaluation_request(pipeline, request, stage, revision)
            started = api.start(api.plan(evaluation_request))
        except Exception:
            return self._finish(stage, StageState.FAILED)
        if started.accepted is not True:
            return self._finish(stage, StageState.FAILED)
        return self._evaluate_stage_outcome(stage, started.run)

    # -- resume planning ----------------------------------------------------------------

    def _replan(self, head: _Head, request: PipelineRequest) -> _Head:
        """Recompute every attempt key; keep matching succeeded and in-flight stages, re-plan the rest."""
        record = head.record
        pipeline = record.pipeline
        stages: list[PipelineStage] = []
        upstream_kept = True
        upstream_outputs: tuple[VerifiedArtifact, ...] = ()
        in_flight = False
        for index, stage in enumerate(record.stages):
            inputs = upstream_outputs if stage.name is PipelineStageName.EVALUATE else ()
            key = stage_attempt_key(pipeline.pipeline_id, stage.name, record.spec_digest, inputs)
            if stage.state is StageState.SUCCEEDED and upstream_kept and key == stage.attempt_key:
                outputs = stage.outputs
                if stage.run is not None:
                    live = self._live_outputs(stage)
                    if live != stage.outputs:
                        # The child's artifacts are the authoritative output. A downstream
                        # stage that already succeeded on the recorded inputs is now a lie.
                        self._refuse_stale_success(record)
                        outputs = live
                if outputs != stage.outputs:
                    stage = PipelineStage(stage.name, stage.state, stage.attempt_key, stage.run, stage.inputs, outputs)
                stages.append(stage)
                upstream_outputs = outputs
                continue
            if stage.state in _IN_FLIGHT and stage.run is not None:
                stages.append(stage)
                in_flight = True
            elif stage.state is StageState.RUNNING and stage.name is PipelineStageName.EVALUATE and upstream_kept and key == stage.attempt_key:
                stages.append(stage)  # interrupted before the child reference was known; its start is idempotent
                in_flight = True
            elif stage.name is PipelineStageName.TRAIN and stage.state in (StageState.FAILED, StageState.CANCELLED):
                stages.append(stage)  # training identity belongs to the training family: not retried here
            else:
                stages.append(_planned_stage(pipeline, record.spec_digest, stage.name))
            upstream_kept = False
            upstream_outputs = ()
        planned = tuple(stages)
        if planned == record.stages:
            if not in_flight:
                raise _error(PipelineOperationCode.RESUME_INELIGIBLE)
            return head
        first = next((i for i, stage in enumerate(planned) if stage.state not in TERMINAL_STAGE_STATES), None)
        if first is None:
            raise _error(PipelineOperationCode.RESUME_INELIGIBLE)
        if planned[first].state is StageState.PLANNED and planned[first].name is PipelineStageName.EVALUATE and first > 0:
            upstream = planned[first - 1]
            if upstream.state is StageState.SUCCEEDED:
                spec = request.evaluate
                if spec is None or sum(1 for item in upstream.outputs if item.role == spec.model_artifact_role) != 1:
                    raise _error(PipelineOperationCode.STAGE_INPUT_MISSING)
        updated = _rebuild(record, planned, revision=head.revision + 1, cancel_requested=False)
        return self._transition(head, updated, "resumed")

    # -- verbs -----------------------------------------------------------------------

    def plan(self, request: PipelineRequest) -> PipelinePlan:
        if request.unadmitted_stages:
            raise _error(PipelineOperationCode.STAGE_UNSUPPORTED)
        if request.evaluate is not None and self._evaluation is None:
            raise _error(PipelineOperationCode.STAGE_UNSUPPORTED)
        try:
            if request.train is not None:
                loaded = self._training.load(request.train.canonical_json)
                if loaded.project_ref != request.project_ref:
                    raise ValueError("train request project does not match the pipeline")
                self._training.plan(self._training.resolve(loaded), request.train.provider)
            if request.evaluate is not None:
                spec = request.evaluate
                self._evaluation_api().plan(EvaluationRequest(
                    f"{request.request_id}/evaluate/plan", request.project_ref,
                    EvaluationModelRef(spec.model_ref, "pending"), spec.backend, spec.scenario_refs,
                    spec.preset, spec.tags, spec.case_limit,
                ))
        except PipelineOperationError:
            raise
        except Exception:
            raise _error(PipelineOperationCode.SPEC_INVALID) from None
        return PipelinePlan(PIPELINE_PLAN_SCHEMA_VERSION, request)

    def start(self, plan: PipelinePlan) -> PipelineStart:
        pipeline = PipelineRef(_pipeline_id(plan), plan.request.project_ref)
        if self._head(pipeline) is not None:
            return PipelineStart(pipeline, True)
        if plan.request.unadmitted_stages or (plan.request.evaluate is not None and self._evaluation is None):
            raise _error(PipelineOperationCode.STAGE_UNSUPPORTED)
        self._store_plan(pipeline, plan)
        genesis = PipelineRecord(
            PIPELINE_RECORD_SCHEMA_VERSION, pipeline, plan.spec_digest, PipelineState.PLANNED, 1,
            tuple(_planned_stage(pipeline, plan.spec_digest, name) for name in plan.request.stages),
        )
        head = self._create(genesis)
        if head is None:
            return PipelineStart(pipeline, True)
        started = PipelineRecord(
            PIPELINE_RECORD_SCHEMA_VERSION, pipeline, plan.spec_digest, PipelineState.RUNNING, 2, genesis.stages,
        )
        self._transition(head, started, "started")
        self._drive(pipeline, plan)
        return PipelineStart(pipeline, True)

    def show(self, pipeline: PipelineRef) -> PipelineRecord:
        return self._require_head(pipeline).record

    def _refuse_stale_success(self, record: PipelineRecord) -> None:
        """A succeeded stage built on inputs its upstream child no longer produces is refused."""
        for index, stage in enumerate(record.stages):
            if stage.state is not StageState.SUCCEEDED or stage.run is None:
                continue
            if any(later.state is StageState.SUCCEEDED for later in record.stages[index + 1:]):
                if self._live_outputs(stage) != stage.outputs:
                    raise _error(PipelineOperationCode.STAGE_DIGEST_MISMATCH)

    def resume(self, pipeline: PipelineRef) -> PipelineRecord:
        head = self._require_head(pipeline)
        if head.record.state in _RESUME_INELIGIBLE:
            if head.record.state is PipelineState.SUCCEEDED:
                self._refuse_stale_success(head.record)
            raise _error(PipelineOperationCode.RESUME_INELIGIBLE)
        with self._lock:
            if pipeline.pipeline_id in self._driving:
                raise _error(PipelineOperationCode.RESUME_INELIGIBLE)
        plan = self._plan_for(head)
        if head.record.state is not PipelineState.CANCEL_REQUESTED:
            self._replan(head, plan.request)
        self._drive(pipeline, plan)
        return self._require_head(pipeline).record

    def cancel(self, pipeline: PipelineRef, reason: str) -> PipelineRecord:
        for _ in range(APPEND_ATTEMPTS):
            head = self._require_head(pipeline)
            record = head.record
            if record.state in TERMINAL_PIPELINE_STATES:
                raise _error(PipelineOperationCode.CANCEL_INELIGIBLE)
            if record.state is PipelineState.CANCEL_REQUESTED:
                return record
            index = next((i for i, stage in enumerate(record.stages) if stage.state in _IN_FLIGHT), None)
            if index is None:
                raise _error(PipelineOperationCode.CANCEL_INELIGIBLE)
            stage = record.stages[index]
            projected = self._cancel_child(stage, reason)
            updated = _with_stage(record, projected, revision=head.revision + 1, cancel_requested=True)
            updated = _rebuild(
                updated, _resolve_later_stages(updated, index, StageState.CANCELLED),
                revision=head.revision + 1, cancel_requested=True,
            )
            try:
                committed = self._transition(head, updated, "cancel_requested")
            except PipelineOperationError as error:
                if error.code is not PipelineOperationCode.STATE_CONFLICT:
                    raise
                continue
            observations = _ObservationWriter(self._streams, pipeline, self._clock)
            for item in committed.record.stages[index:]:
                observations.completed(item)
            return committed.record
        raise _error(PipelineOperationCode.STATE_CONFLICT)

    def _cancel_child(self, stage: PipelineStage, reason: str) -> PipelineStage:
        """Request cancellation on the one in-flight child; project its answer onto the stage."""
        if stage.run is None:
            return stage  # the evaluate child has no reference yet; the driver finalizes on return
        if type(stage.run) is TrainingRunRef:
            try:
                self._runs.cancel(stage.run, reason)
            except Exception:
                pass  # the child answers for itself below; a terminal child is simply observed
            return self._train_stage_outcome(stage, poll=False)
        try:
            self._evaluation_api().cancel(stage.run, reason)  # type: ignore[arg-type]
        except Exception:
            pass
        return self._evaluate_stage_outcome(stage, stage.run)  # type: ignore[arg-type]

    def list(self, request: PipelineListRequest) -> PipelinePage:
        prefix = f"pipeline/{_project_digest(request.project_ref)}/"
        after_key = None
        if request.cursor is not None:
            try:
                digest_text(request.cursor, "cursor")
            except ValueError:
                raise _error(PipelineOperationCode.CURSOR_INVALID) from None
            after_key = prefix + request.cursor
            if self._read_stored(after_key) is None:
                raise _error(PipelineOperationCode.CURSOR_INVALID)
        try:
            page = self._records.list_page(partition=_PIPELINE, prefix=prefix, after_key=after_key, limit=request.limit)
        except Exception:
            raise _integrity() from None
        records = []
        for stored in page.records:
            if type(stored) is not StoredRecordV1 or not stored.key.startswith(prefix):
                raise _integrity()
            records.append(self._validate(stored.key, stored).record)
        next_cursor = None
        if page.truncated:
            if type(page.next_cursor) is not str or not page.next_cursor.startswith(prefix):
                raise _integrity()
            next_cursor = page.next_cursor[len(prefix):]
        return PipelinePage(request, tuple(records), next_cursor, page.truncated is True)

    def observations(self, request: ObservationsRequest) -> ObservationPage:
        pipeline = PipelineRef(request.stream.entity_id, request.stream.project_ref)
        self._require_head(pipeline)
        page = _read_page(self._streams, _OBSERVATION, _observation_key(pipeline), request.after_sequence, request.limit)
        records = []
        for entry in page.entries:
            try:
                record = ObservationRecordV1.from_dict(json.loads(entry.canonical.decode("utf-8")))
            except Exception:
                raise _integrity() from None
            if record.sequence != entry.sequence or record.stream != request.stream:
                raise _integrity()
            records.append(record)
        return ObservationPage(request, tuple(records), page.next_cursor, page.truncated is True)


def compose_reference_pipelines(
    *,
    records: DurableRecordStorePort,
    streams: DurableStreamStorePort,
    clock,
    training,
    runs,
    evaluation,
    pipelines: ReferencePipelinePortsV1,
) -> ReferencePipelineOperationsV1:
    """The ``PipelinesOperations`` implementation over the host stores and the child families."""
    return ReferencePipelineOperationsV1(
        records=records, streams=streams, clock=clock, training=training, runs=runs,
        evaluation=evaluation, ports=pipelines,
    )


__all__ = [
    "DEFAULT_MAXIMUM_CHILD_POLLS",
    "PIPELINE_EVENT_BUDGET_BASE",
    "PIPELINE_EVENT_BUDGET_PER_STAGE",
    "PIPELINE_HEAD_MAXIMUM_BYTES",
    "PipelinePacingPort",
    "ReferencePipelineOperationsV1",
    "ReferencePipelinePortsV1",
    "compose_reference_pipelines",
    "pipeline_event_budget",
]
