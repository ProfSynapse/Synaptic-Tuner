"""Engine adapters satisfying the coordinator store ports over ``DurableRecordStorePort``.

Location: ``synaptic_tuner/api/v1/reference/repositories.py``.

The coordinator (``tuner/execution/coordinator_v1``) persists through five
internal store ports (``PlanningStorePortV1``, ``WorkflowStorePortV1``,
``PreparationStorePortV1``, ``ExecutionGrantStorePortV1``,
``ReconciliationGrantStorePortV1``). Those ports traffic in ``foundation_v2``
types and are never public. A host implements only the public
``DurableRecordStorePort`` (opaque canonical bytes, compare-and-swap); the
five classes here translate between the two, so the same host store serves
every partition and the record shapes stay an engine concern behind
``schema_version``.

Every adapter re-derives and re-validates what it reads: a decoded workflow
record must rebuild to the same ``record_digest`` the head document recorded,
every historical transition must replay to the following record, and grants
must authenticate against the authority that issued them. The validators are
the module-level functions of ``coordinator_v1/stores.py`` (``revalidate_*``,
``replay_transition``, ``validate_*``), shared with the in-memory stores so a
durable store cannot drift from the in-memory one. Any decode or integrity
failure surfaces as ``CoordinatorStoreError(INTEGRITY_ERROR)``; nothing is
ever silently accepted.

Record layout (partition -> key -> document):

- ``workflow`` / ``run/<project_digest>/<run_key>``: one atomic head document
  per run holding the full record history, its digests, the recorded
  transitions and their digests. One compare-and-swap advances a run, which
  keeps history and head consistent without a second write.
- ``workflow`` / ``plan/<project_digest>/<plan_fingerprint>``: plan index
  naming the run that owns the plan (at most one run per plan and project).
- ``plan`` / ``<plan_fingerprint>`` and ``plan_context`` /
  ``<provider_context_digest>``: public planning contracts, stored through
  their own ``to_dict``/``from_dict`` codecs.
- ``preparation`` / ``<preparation_digest>``: ``CanonicalPreparationV2`` bytes.
- ``execution_grant`` and ``reconciliation_grant`` / ``<slot digest>``:
  authenticated grant canonical bytes keyed by the coordinator's slot.

Composed by ``compose_reference_stores`` in this module and consumed by
``synaptic_tuner/api/v1/reference/__init__.py``. Imports no
``api/v1/persistence.py`` and no ``ProjectContext``.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import json
from threading import RLock
import types
import typing

from synaptic_tuner.api.v1.planning import ProviderPlanContextV1, TrainingPlan
from synaptic_tuner.api.v1.ports import DurableRecordStorePort, StoragePartition, StoredRecordV1
from synaptic_tuner.api.v1.results import TrainingRunRef
from tuner.execution.coordinator_v1.coordinator import (
    ApplyArtifactVerificationTransitionV1,
    ApplyCancelEffectTransitionV1,
    ApplyProviderObservationTransitionV1,
    ApplyReverificationTransitionV1,
    ApplyStageEffectTransitionV1,
    ApplySubmitEffectTransitionV1,
    BeginPreparationTransitionV1,
    CoordinatorTransitionKindV1,
    CoordinatorTransitionV1,
    ExecutionGrantSlotV1,
    ReconciliationGrantSlotV1,
    RecordCancelIntentTransitionV1,
    RecordStageIntentTransitionV1,
    RecordSubmitIntentTransitionV1,
)
from tuner.execution.coordinator_v1.model import (
    WorkflowPhaseV1,
    WorkflowRecordV1,
    WorkflowStorePageV1,
)
from tuner.execution.coordinator_v1.stores import (
    CoordinatorStoreCode,
    CoordinatorStoreError,
    replay_transition,
    revalidate_transition,
    revalidate_workflow,
    validate_execution_grant,
    validate_preparation,
    validate_reconciliation_grant,
    workflow_run_key,
)
from tuner.execution.foundation_v2.authority import (
    AuthenticatedGrantV2,
    AuthenticatedReconciliationGrantV1,
    GrantContentV2,
    ReconciliationGrantContentV1,
)
from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    digest_text,
    domain_digest,
    parse_canonical_object,
    safe_ref,
)
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.foundation_v2.receipts import AuthenticatedReceiptV2
from tuner.execution.foundation_v2.repository import EffectRecordV2


WORKFLOW_HEAD_SCHEMA = "synaptic-reference-workflow-head/v1"
WORKFLOW_PLAN_INDEX_SCHEMA = "synaptic-reference-workflow-plan-index/v1"
MAX_DOCUMENT_BYTES = 16 * 1024 * 1024

_WORKFLOW = StoragePartition.WORKFLOW.value
_PLAN = StoragePartition.PLAN.value
_PLAN_CONTEXT = StoragePartition.PLAN_CONTEXT.value
_PREPARATION = StoragePartition.PREPARATION.value
_EXECUTION_GRANT = StoragePartition.EXECUTION_GRANT.value
_RECONCILIATION_GRANT = StoragePartition.RECONCILIATION_GRANT.value


def _closed(code: CoordinatorStoreCode) -> CoordinatorStoreError:
    return CoordinatorStoreError(code)


def _integrity() -> CoordinatorStoreError:
    return _closed(CoordinatorStoreCode.INTEGRITY_ERROR)


# --------------------------------------------------------------------------
# Reflective codec for the coordinator's frozen dataclass graph.
#
# Every type reachable from WorkflowRecordV1, the transition dataclasses and
# the authenticated grants is a frozen dataclass whose fields are str, int,
# bool, bytes, a str Enum, an optional, a homogeneous tuple or another such
# dataclass. Two EffectRecordV2 fields are annotated loosely (``grant:
# object``, ``results: tuple``) and are pinned here to the exact types the
# foundation validators demand. CanonicalPreparationV2 is the one non-dataclass
# leaf and travels as its own canonical bytes.
# --------------------------------------------------------------------------

_FIELD_TYPE_OVERRIDES: dict[tuple[type, str], object] = {
    (EffectRecordV2, "grant"): AuthenticatedGrantV2,
    (EffectRecordV2, "results"): tuple[AuthenticatedReceiptV2, ...],
}
_HINT_CACHE: dict[type, dict[str, object]] = {}


def _hints(cls: type) -> dict[str, object]:
    cached = _HINT_CACHE.get(cls)
    if cached is None:
        resolved = typing.get_type_hints(cls)
        cached = {
            field.name: _FIELD_TYPE_OVERRIDES.get((cls, field.name), resolved[field.name])
            for field in fields(cls)
        }
        _HINT_CACHE[cls] = cached
    return cached


def _union_members(hint: object) -> tuple[object, ...] | None:
    if typing.get_origin(hint) in (types.UnionType, typing.Union):
        return typing.get_args(hint)
    return None


def _encode(value: object, hint: object) -> object:
    members = _union_members(hint)
    if members is not None:
        if value is None and type(None) in members:
            return None
        for member in members:
            if member is type(None):
                continue
            if _is_instance_of(value, member):
                return _encode(value, member)
        raise _integrity()
    if typing.get_origin(hint) is tuple:
        (item_hint, ellipsis) = typing.get_args(hint)
        if ellipsis is not Ellipsis or type(value) is not tuple:
            raise _integrity()
        return [_encode(item, item_hint) for item in value]
    if hint is bytes:
        if type(value) is not bytes:
            raise _integrity()
        return value.hex()
    if hint in (str, int, bool):
        if type(value) is not hint:
            raise _integrity()
        return value
    if hint is CanonicalPreparationV2:
        if type(value) is not CanonicalPreparationV2:
            raise _integrity()
        return value.canonical_bytes.hex()
    if isinstance(hint, type) and issubclass(hint, Enum):
        if type(value) is not hint:
            raise _integrity()
        return value.value
    if isinstance(hint, type) and is_dataclass(hint):
        if type(value) is not hint:
            raise _integrity()
        return {
            name: _encode(getattr(value, name), field_hint)
            for name, field_hint in _hints(hint).items()
        }
    raise _integrity()


def _is_instance_of(value: object, hint: object) -> bool:
    if typing.get_origin(hint) is tuple:
        return type(value) is tuple
    return isinstance(hint, type) and type(value) is hint


def _decode(document: object, hint: object) -> object:
    members = _union_members(hint)
    if members is not None:
        if document is None:
            if type(None) in members:
                return None
            raise _integrity()
        concrete = [member for member in members if member is not type(None)]
        if len(concrete) != 1:
            raise _integrity()
        return _decode(document, concrete[0])
    if typing.get_origin(hint) is tuple:
        (item_hint, ellipsis) = typing.get_args(hint)
        if ellipsis is not Ellipsis or type(document) is not list:
            raise _integrity()
        return tuple(_decode(item, item_hint) for item in document)
    if hint is bytes:
        if type(document) is not str:
            raise _integrity()
        try:
            return bytes.fromhex(document)
        except ValueError:
            raise _integrity() from None
    if hint in (str, int, bool):
        if type(document) is not hint:
            raise _integrity()
        return document
    if hint is CanonicalPreparationV2:
        if type(document) is not str:
            raise _integrity()
        try:
            return CanonicalPreparationV2.parse(bytes.fromhex(document))
        except Exception:
            raise _integrity() from None
    if isinstance(hint, type) and issubclass(hint, Enum):
        if type(document) is not str:
            raise _integrity()
        try:
            return hint(document)
        except ValueError:
            raise _integrity() from None
    if isinstance(hint, type) and is_dataclass(hint):
        field_hints = _hints(hint)
        if type(document) is not dict or set(document) != set(field_hints):
            raise _integrity()
        try:
            return hint(
                **{name: _decode(document[name], field_hint) for name, field_hint in field_hints.items()}
            )
        except CoordinatorStoreError:
            raise
        except Exception:
            raise _integrity() from None
    raise _integrity()


def _dump(document: dict[str, object]) -> bytes:
    try:
        encoded = json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise _integrity() from None
    if len(encoded) > MAX_DOCUMENT_BYTES:
        raise _integrity()
    return encoded


def _load(raw: bytes, schema_version: str) -> dict[str, object]:
    if type(raw) is not bytes or not raw or len(raw) > MAX_DOCUMENT_BYTES:
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


_TRANSITION_TYPES: dict[str, type] = {
    CoordinatorTransitionKindV1.BEGIN_PREPARATION.value: BeginPreparationTransitionV1,
    CoordinatorTransitionKindV1.RECORD_STAGE_INTENT.value: RecordStageIntentTransitionV1,
    CoordinatorTransitionKindV1.APPLY_STAGE_EFFECT.value: ApplyStageEffectTransitionV1,
    CoordinatorTransitionKindV1.RECORD_SUBMIT_INTENT.value: RecordSubmitIntentTransitionV1,
    CoordinatorTransitionKindV1.APPLY_SUBMIT_EFFECT.value: ApplySubmitEffectTransitionV1,
    CoordinatorTransitionKindV1.RECORD_CANCEL_INTENT.value: RecordCancelIntentTransitionV1,
    CoordinatorTransitionKindV1.APPLY_CANCEL_EFFECT.value: ApplyCancelEffectTransitionV1,
    CoordinatorTransitionKindV1.APPLY_PROVIDER_OBSERVATION.value: ApplyProviderObservationTransitionV1,
    CoordinatorTransitionKindV1.APPLY_ARTIFACT_VERIFICATION.value: ApplyArtifactVerificationTransitionV1,
    CoordinatorTransitionKindV1.APPLY_REVERIFICATION.value: ApplyReverificationTransitionV1,
}


def encode_workflow_record(record: WorkflowRecordV1) -> dict[str, object]:
    """Encode a validated workflow record as a JSON-compatible document."""
    return _encode(revalidate_workflow(record), WorkflowRecordV1)


def decode_workflow_record(document: object) -> WorkflowRecordV1:
    """Decode and re-validate a workflow record document."""
    return revalidate_workflow(_decode(document, WorkflowRecordV1))


def encode_transition(transition: CoordinatorTransitionV1) -> dict[str, object]:
    rebuilt, _ = revalidate_transition(transition)
    return {"kind": rebuilt.kind.value, "value": _encode(rebuilt, type(rebuilt))}


def decode_transition(document: object) -> CoordinatorTransitionV1:
    if type(document) is not dict or set(document) != {"kind", "value"}:
        raise _integrity()
    transition_type = _TRANSITION_TYPES.get(document["kind"])
    if transition_type is None:
        raise _integrity()
    rebuilt, _ = revalidate_transition(_decode(document["value"], transition_type))
    return rebuilt


def _decode_authenticated(raw: bytes, *, schema_version: str, content_type: type, envelope: type):
    try:
        document = parse_canonical_object(raw, name="authenticated grant")
    except Exception:
        raise _integrity() from None
    if set(document) != {"schema_version", "content", "authority_ref", "tag"}:
        raise _integrity()
    if document["schema_version"] != schema_version:
        raise _integrity()
    content = _decode(document["content"], content_type)
    try:
        grant = envelope(content, document["authority_ref"], document["tag"])
    except Exception:
        raise _integrity() from None
    if grant.canonical_bytes != raw:
        raise _integrity()
    return grant


def decode_execution_grant(raw: bytes) -> AuthenticatedGrantV2:
    return _decode_authenticated(
        raw,
        schema_version="synaptic-authenticated-grant/v3",
        content_type=GrantContentV2,
        envelope=AuthenticatedGrantV2,
    )


def decode_reconciliation_grant(raw: bytes) -> AuthenticatedReconciliationGrantV1:
    return _decode_authenticated(
        raw,
        schema_version="synaptic-authenticated-reconciliation-grant/v2",
        content_type=ReconciliationGrantContentV1,
        envelope=AuthenticatedReconciliationGrantV1,
    )


# --------------------------------------------------------------------------
# Store adapters
# --------------------------------------------------------------------------


def _read(records: DurableRecordStorePort, partition: str, key: str) -> StoredRecordV1 | None:
    try:
        stored = records.read(partition=partition, key=key)
    except Exception:
        raise _integrity() from None
    if stored is None:
        return None
    if type(stored) is not StoredRecordV1 or stored.key != key:
        raise _integrity()
    return stored


def _project_digest(project_ref: str) -> str:
    try:
        safe_ref(project_ref, "project_ref")
    except Exception:
        raise _closed(CoordinatorStoreCode.BINDING_MISMATCH) from None
    return domain_digest("synaptic-reference-project-key/v1", project_ref.encode("utf-8"))


class DurableWorkflowStoreV1:
    """``WorkflowStorePortV1`` over one head document per run."""

    def __init__(
        self,
        records: DurableRecordStorePort,
        *,
        foundation_authenticator,
        assessment_authenticator,
        observation_authenticator,
        artifact_verifier,
    ) -> None:
        self._records = records
        self._foundation_authenticator = foundation_authenticator
        self._assessment_authenticator = assessment_authenticator
        self._observation_authenticator = observation_authenticator
        self._artifact_verifier = artifact_verifier
        self._lock = RLock()

    # -- keys and documents ------------------------------------------------

    @staticmethod
    def _head_key(run: TrainingRunRef) -> str:
        if type(run) is not TrainingRunRef:
            raise _closed(CoordinatorStoreCode.BINDING_MISMATCH)
        return f"run/{_project_digest(run.project_ref)}/{workflow_run_key(run)}"

    @staticmethod
    def _plan_key(project_ref: str, plan_fingerprint: str) -> str:
        try:
            digest_text(plan_fingerprint, "plan_fingerprint")
        except Exception:
            raise _closed(CoordinatorStoreCode.BINDING_MISMATCH) from None
        return f"plan/{_project_digest(project_ref)}/{plan_fingerprint}"

    @staticmethod
    def _index_document(run: TrainingRunRef) -> bytes:
        return _dump({"schema_version": WORKFLOW_PLAN_INDEX_SCHEMA, "run": run.to_dict()})

    @staticmethod
    def _decode_index(raw: bytes) -> TrainingRunRef:
        document = _load(raw, WORKFLOW_PLAN_INDEX_SCHEMA)
        if set(document) != {"schema_version", "run"}:
            raise _integrity()
        try:
            return TrainingRunRef.from_dict(document["run"])
        except Exception:
            raise _integrity() from None

    @staticmethod
    def _head_document(
        history: tuple[WorkflowRecordV1, ...],
        transitions: tuple[CoordinatorTransitionV1, ...],
    ) -> bytes:
        digests = []
        encoded_transitions = []
        for transition in transitions:
            _, fingerprint = revalidate_transition(transition)
            digests.append(fingerprint)
            encoded_transitions.append(encode_transition(transition))
        return _dump(
            {
                "schema_version": WORKFLOW_HEAD_SCHEMA,
                "history": [encode_workflow_record(record) for record in history],
                "history_digests": [record.record_digest for record in history],
                "transitions": encoded_transitions,
                "transition_digests": digests,
            }
        )

    def _replay(self, current: WorkflowRecordV1, transition: CoordinatorTransitionV1) -> WorkflowRecordV1:
        return replay_transition(
            current,
            transition,
            foundation_authenticator=self._foundation_authenticator,
            assessment_authenticator=self._assessment_authenticator,
            observation_authenticator=self._observation_authenticator,
            artifact_verifier=self._artifact_verifier,
        )

    def _validated_history(
        self, raw: bytes, key: str
    ) -> tuple[tuple[WorkflowRecordV1, ...], tuple[CoordinatorTransitionV1, ...]]:
        """Decode a head document and prove every recorded step re-derives."""
        document = _load(raw, WORKFLOW_HEAD_SCHEMA)
        expected_keys = {"schema_version", "history", "history_digests", "transitions", "transition_digests"}
        if set(document) != expected_keys:
            raise _integrity()
        history_documents = document["history"]
        history_digests = document["history_digests"]
        transition_documents = document["transitions"]
        transition_digests = document["transition_digests"]
        if (
            type(history_documents) is not list
            or type(history_digests) is not list
            or type(transition_documents) is not list
            or type(transition_digests) is not list
            or not history_documents
            or len(history_digests) != len(history_documents)
            or len(transition_documents) != len(history_documents) - 1
            or len(transition_digests) != len(transition_documents)
        ):
            raise _integrity()
        try:
            genesis = decode_workflow_record(history_documents[0])
            if (
                genesis.phase is not WorkflowPhaseV1.PLANNED
                or genesis.revision != 0
                or self._head_key(genesis.run) != key
                or history_digests[0] != genesis.record_digest
            ):
                raise _integrity()
            history = [genesis]
            transitions = []
            previous = genesis
            for index, transition_document in enumerate(transition_documents):
                transition = decode_transition(transition_document)
                _, fingerprint = revalidate_transition(transition)
                if transition_digests[index] != fingerprint:
                    raise _integrity()
                following = decode_workflow_record(history_documents[index + 1])
                if (
                    following.revision != previous.revision + 1
                    or following.run != genesis.run
                    or following.plan_fingerprint != genesis.plan_fingerprint
                    or history_digests[index + 1] != following.record_digest
                ):
                    raise _integrity()
                replayed = self._replay(previous, transition)
                if replayed != following or replayed.record_digest != following.record_digest:
                    raise _integrity()
                history.append(following)
                transitions.append(transition)
                previous = following
        except CoordinatorStoreError:
            raise _integrity() from None
        except Exception:
            raise _integrity() from None
        return tuple(history), tuple(transitions)

    def _retained(self, stored: StoredRecordV1, key: str) -> WorkflowRecordV1:
        history, _ = self._validated_history(stored.canonical, key)
        retained = history[-1]
        index = _read(self._records, _WORKFLOW, self._plan_key(retained.run.project_ref, retained.plan_fingerprint))
        if index is None or self._decode_index(index.canonical) != retained.run:
            raise _integrity()
        return retained

    # -- WorkflowStorePortV1 ---------------------------------------------

    def create(self, record: WorkflowRecordV1) -> bool:
        candidate = revalidate_workflow(record)
        if candidate.phase is not WorkflowPhaseV1.PLANNED or candidate.revision != 0:
            raise _closed(CoordinatorStoreCode.TRANSITION_INVALID)
        key = self._head_key(candidate.run)
        plan_key = self._plan_key(candidate.run.project_ref, candidate.plan_fingerprint)
        with self._lock:
            try:
                indexed = self._records.put_if_absent(
                    partition=_WORKFLOW, key=plan_key, canonical=self._index_document(candidate.run)
                )
            except Exception:
                raise _integrity() from None
            if indexed is not True:
                index = _read(self._records, _WORKFLOW, plan_key)
                if index is None:
                    raise _integrity()
                owner = self._decode_index(index.canonical)
                if owner != candidate.run:
                    other = _read(self._records, _WORKFLOW, self._head_key(owner))
                    if other is None:
                        raise _integrity()
                    retained = self._retained(other, self._head_key(owner))
                    if retained.plan_fingerprint != candidate.plan_fingerprint:
                        raise _integrity()
                    raise _closed(CoordinatorStoreCode.CONFLICT)
            try:
                created = self._records.create(
                    partition=_WORKFLOW, key=key, canonical=self._head_document((candidate,), ())
                )
            except Exception:
                raise _integrity() from None
            if created is True:
                return True
            stored = _read(self._records, _WORKFLOW, key)
            if stored is None:
                raise _integrity()
            existing = self._retained(stored, key)
            if existing == candidate and existing.record_digest == candidate.record_digest:
                return False
            raise _closed(CoordinatorStoreCode.CONFLICT)

    def get(self, run: TrainingRunRef) -> WorkflowRecordV1 | None:
        key = self._head_key(run)
        stored = _read(self._records, _WORKFLOW, key)
        if stored is None:
            return None
        retained = self._retained(stored, key)
        if retained.run != run:
            raise _integrity()
        return retained

    def get_by_plan(self, project_ref: str, plan_fingerprint: str) -> WorkflowRecordV1 | None:
        plan_key = self._plan_key(project_ref, plan_fingerprint)
        index = _read(self._records, _WORKFLOW, plan_key)
        if index is None:
            return None
        owner = self._decode_index(index.canonical)
        if owner.project_ref != project_ref:
            raise _integrity()
        key = self._head_key(owner)
        stored = _read(self._records, _WORKFLOW, key)
        if stored is None:
            raise _integrity()
        retained = self._retained(stored, key)
        if retained.run != owner or retained.plan_fingerprint != plan_fingerprint:
            raise _integrity()
        return retained

    def list_page(self, project_ref: str, *, after_run_key: str | None, limit: int) -> WorkflowStorePageV1:
        try:
            safe_ref(project_ref, "project_ref")
            if after_run_key is not None:
                digest_text(after_run_key, "after_run_key")
            if type(limit) is not int or not 1 <= limit <= 100:
                raise ValueError("limit invalid")
        except Exception:
            raise _closed(CoordinatorStoreCode.BINDING_MISMATCH) from None
        prefix = f"run/{_project_digest(project_ref)}/"
        after_key = None
        if after_run_key is not None:
            after_key = prefix + after_run_key
            if _read(self._records, _WORKFLOW, after_key) is None:
                raise _closed(CoordinatorStoreCode.BINDING_MISMATCH)
        try:
            page = self._records.list_page(
                partition=_WORKFLOW, prefix=prefix, after_key=after_key, limit=limit
            )
        except Exception:
            raise _integrity() from None
        if type(page.truncated) is not bool:
            raise _integrity()
        selected = []
        for stored in page.records:
            if type(stored) is not StoredRecordV1 or not stored.key.startswith(prefix):
                raise _integrity()
            retained = self._retained(stored, stored.key)
            if (
                retained.run.project_ref != project_ref
                or stored.key != prefix + workflow_run_key(retained.run)
            ):
                raise _integrity()
            selected.append(retained)
        return WorkflowStorePageV1(records=tuple(selected), has_more=page.truncated)

    def is_descendant(self, ancestor: WorkflowRecordV1, descendant: WorkflowRecordV1) -> bool:
        ancestor = revalidate_workflow(ancestor)
        descendant = revalidate_workflow(descendant)
        key = self._head_key(ancestor.run)
        if self._head_key(descendant.run) != key:
            return False
        stored = _read(self._records, _WORKFLOW, key)
        if stored is None:
            return False
        history, _ = self._validated_history(stored.canonical, key)
        if ancestor.revision >= descendant.revision or descendant.revision >= len(history):
            return False
        stored_ancestor = history[ancestor.revision]
        stored_descendant = history[descendant.revision]
        return bool(
            stored_ancestor == ancestor
            and stored_ancestor.record_digest == ancestor.record_digest
            and stored_descendant == descendant
            and stored_descendant.record_digest == descendant.record_digest
        )

    def compare_and_swap(
        self,
        expected: WorkflowRecordV1,
        replacement: WorkflowRecordV1,
        *,
        transition: CoordinatorTransitionV1,
    ) -> bool:
        expected = revalidate_workflow(expected)
        replacement = revalidate_workflow(replacement)
        key = self._head_key(expected.run)
        if self._head_key(replacement.run) != key:
            raise _closed(CoordinatorStoreCode.BINDING_MISMATCH)
        if replacement.revision != expected.revision + 1:
            raise _closed(CoordinatorStoreCode.TRANSITION_INVALID)
        with self._lock:
            stored = _read(self._records, _WORKFLOW, key)
            if stored is None:
                return False
            history, transitions = self._validated_history(stored.canonical, key)
            retained = history[-1]
            if retained.revision != expected.revision:
                return False
            if retained.record_digest != expected.record_digest or retained != expected:
                raise _closed(CoordinatorStoreCode.CONFLICT)
            replayed = self._replay(retained, transition)
            if replayed.revision != retained.revision + 1 or replayed != replacement:
                raise _closed(CoordinatorStoreCode.TRANSITION_INVALID)
            transition, _ = revalidate_transition(transition)
            document = self._head_document(history + (replacement,), transitions + (transition,))
            try:
                swapped = self._records.compare_and_swap(
                    partition=_WORKFLOW, key=key, expected_revision=stored.revision, canonical=document
                )
            except Exception:
                raise _integrity() from None
            if type(swapped) is not bool:
                raise _integrity()
            return swapped


class DurablePlanningStoreV1:
    """``PlanningStorePortV1`` over the ``plan`` and ``plan_context`` partitions."""

    def __init__(self, records: DurableRecordStorePort) -> None:
        self._records = records

    @staticmethod
    def _plan(value: TrainingPlan) -> TrainingPlan:
        if type(value) is not TrainingPlan:
            raise _closed(CoordinatorStoreCode.BINDING_MISMATCH)
        try:
            return TrainingPlan.from_dict(value.to_dict())
        except Exception:
            raise _integrity() from None

    @staticmethod
    def _context(value: ProviderPlanContextV1) -> ProviderPlanContextV1:
        if type(value) is not ProviderPlanContextV1:
            raise _closed(CoordinatorStoreCode.BINDING_MISMATCH)
        try:
            return ProviderPlanContextV1.from_dict(value.to_dict())
        except Exception:
            raise _integrity() from None

    def _put(self, partition: str, key: str, canonical: bytes) -> bool:
        try:
            admitted = self._records.put_if_absent(partition=partition, key=key, canonical=canonical)
        except Exception:
            raise _integrity() from None
        if admitted is True:
            return True
        stored = _read(self._records, partition, key)
        if stored is None:
            raise _integrity()
        if stored.canonical == canonical:
            return False
        raise _closed(CoordinatorStoreCode.CONFLICT)

    def put_plan_if_absent(self, plan: TrainingPlan) -> bool:
        owned = self._plan(plan)
        return self._put(_PLAN, owned.plan_fingerprint, canonical_bytes(owned.to_dict()))

    def get_plan(self, plan_fingerprint: str) -> TrainingPlan | None:
        try:
            digest_text(plan_fingerprint, "plan_fingerprint")
        except Exception:
            raise _closed(CoordinatorStoreCode.BINDING_MISMATCH) from None
        stored = _read(self._records, _PLAN, plan_fingerprint)
        if stored is None:
            return None
        try:
            plan = TrainingPlan.from_dict(parse_canonical_object(stored.canonical, name="training plan"))
        except Exception:
            raise _integrity() from None
        if plan.plan_fingerprint != plan_fingerprint or canonical_bytes(plan.to_dict()) != stored.canonical:
            raise _integrity()
        return plan

    def put_context_if_absent(self, context: ProviderPlanContextV1) -> bool:
        owned = self._context(context)
        return self._put(_PLAN_CONTEXT, owned.provider_context_digest, canonical_bytes(owned.to_dict()))

    def get_context(self, context_digest: str) -> ProviderPlanContextV1 | None:
        try:
            digest_text(context_digest, "context_digest")
        except Exception:
            raise _closed(CoordinatorStoreCode.BINDING_MISMATCH) from None
        stored = _read(self._records, _PLAN_CONTEXT, context_digest)
        if stored is None:
            return None
        try:
            context = ProviderPlanContextV1.from_dict(
                parse_canonical_object(stored.canonical, name="provider plan context")
            )
        except Exception:
            raise _integrity() from None
        if (
            context.provider_context_digest != context_digest
            or canonical_bytes(context.to_dict()) != stored.canonical
        ):
            raise _integrity()
        return context


class DurablePreparationStoreV1:
    """``PreparationStorePortV1`` over the ``preparation`` partition."""

    def __init__(self, records: DurableRecordStorePort) -> None:
        self._records = records

    @staticmethod
    def _parse(raw: bytes) -> CanonicalPreparationV2:
        try:
            parsed = CanonicalPreparationV2.parse(raw)
        except Exception:
            raise _integrity() from None
        parsed = validate_preparation(parsed)
        if parsed.canonical_bytes != raw:
            raise _integrity()
        return parsed

    def put_if_absent(self, preparation: CanonicalPreparationV2) -> bool:
        candidate = validate_preparation(preparation)
        key = candidate.preparation_digest
        try:
            admitted = self._records.put_if_absent(
                partition=_PREPARATION, key=key, canonical=candidate.canonical_bytes
            )
        except Exception:
            raise _integrity() from None
        if admitted is True:
            return True
        stored = _read(self._records, _PREPARATION, key)
        if stored is None:
            raise _integrity()
        if self._parse(stored.canonical) == candidate:
            return False
        raise _closed(CoordinatorStoreCode.CONFLICT)

    def get(self, preparation_digest: str) -> CanonicalPreparationV2 | None:
        try:
            digest_text(preparation_digest, "preparation_digest")
        except Exception:
            raise _closed(CoordinatorStoreCode.BINDING_MISMATCH) from None
        stored = _read(self._records, _PREPARATION, preparation_digest)
        if stored is None:
            return None
        retained = self._parse(stored.canonical)
        if retained.preparation_digest != preparation_digest:
            raise _closed(CoordinatorStoreCode.CONFLICT)
        return retained


def _slot_key(domain: str, slot) -> str:
    return domain_digest(
        domain, canonical_bytes({field.name: getattr(slot, field.name) for field in fields(slot)})
    )


class DurableExecutionGrantStoreV1:
    """``ExecutionGrantStorePortV1`` over the ``execution_grant`` partition."""

    def __init__(self, records: DurableRecordStorePort, authenticator) -> None:
        self._records = records
        self._authenticator = authenticator

    def _key(self, slot) -> str:
        if type(slot) is not ExecutionGrantSlotV1:
            raise _closed(CoordinatorStoreCode.BINDING_MISMATCH)
        return _slot_key("synaptic-reference-execution-grant-slot/v1", slot)

    def put_if_absent(self, slot, grant, command_bytes) -> bool:
        candidate = validate_execution_grant(self._authenticator, slot, grant, command_bytes)
        key = self._key(slot)
        try:
            admitted = self._records.put_if_absent(
                partition=_EXECUTION_GRANT, key=key, canonical=candidate.canonical_bytes
            )
        except Exception:
            raise _integrity() from None
        if admitted is True:
            return True
        stored = _read(self._records, _EXECUTION_GRANT, key)
        if stored is None:
            raise _integrity()
        existing = validate_execution_grant(
            self._authenticator, slot, decode_execution_grant(stored.canonical), command_bytes
        )
        if existing == candidate and existing.canonical_bytes == candidate.canonical_bytes:
            return False
        raise _closed(CoordinatorStoreCode.CONFLICT)

    def get(self, slot, command_bytes):
        key = self._key(slot)
        stored = _read(self._records, _EXECUTION_GRANT, key)
        if stored is None:
            return None
        return validate_execution_grant(
            self._authenticator, slot, decode_execution_grant(stored.canonical), command_bytes
        )


class DurableReconciliationGrantStoreV1:
    """``ReconciliationGrantStorePortV1`` over the ``reconciliation_grant`` partition."""

    def __init__(self, records: DurableRecordStorePort, authenticator) -> None:
        self._records = records
        self._authenticator = authenticator

    def _key(self, slot) -> str:
        if type(slot) is not ReconciliationGrantSlotV1:
            raise _closed(CoordinatorStoreCode.BINDING_MISMATCH)
        return _slot_key("synaptic-reference-reconciliation-grant-slot/v1", slot)

    def put_if_absent(self, slot, grant, command_bytes, record) -> bool:
        candidate = validate_reconciliation_grant(
            self._authenticator, slot, grant, command_bytes, record
        )
        key = self._key(slot)
        try:
            admitted = self._records.put_if_absent(
                partition=_RECONCILIATION_GRANT, key=key, canonical=candidate.canonical_bytes
            )
        except Exception:
            raise _integrity() from None
        if admitted is True:
            return True
        stored = _read(self._records, _RECONCILIATION_GRANT, key)
        if stored is None:
            raise _integrity()
        existing = validate_reconciliation_grant(
            self._authenticator,
            slot,
            decode_reconciliation_grant(stored.canonical),
            command_bytes,
            record,
        )
        if existing == candidate and existing.canonical_bytes == candidate.canonical_bytes:
            return False
        raise _closed(CoordinatorStoreCode.CONFLICT)

    def get(self, slot, *, command_bytes, record):
        key = self._key(slot)
        stored = _read(self._records, _RECONCILIATION_GRANT, key)
        if stored is None:
            return None
        return validate_reconciliation_grant(
            self._authenticator,
            slot,
            decode_reconciliation_grant(stored.canonical),
            command_bytes,
            record,
        )


__all__ = [
    "DurableExecutionGrantStoreV1",
    "DurablePlanningStoreV1",
    "DurablePreparationStoreV1",
    "DurableReconciliationGrantStoreV1",
    "DurableWorkflowStoreV1",
    "MAX_DOCUMENT_BYTES",
    "WORKFLOW_HEAD_SCHEMA",
    "WORKFLOW_PLAN_INDEX_SCHEMA",
    "decode_execution_grant",
    "decode_reconciliation_grant",
    "decode_transition",
    "decode_workflow_record",
    "encode_transition",
    "encode_workflow_record",
]
