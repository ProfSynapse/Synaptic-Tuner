"""Engine adapters satisfying the coordinator store ports over ``DurableRecordStorePort``.

Location: ``synaptic_tuner/api/v1/reference/repositories.py``.

The coordinator (``tuner/execution/coordinator_v1``) persists through five
internal store ports plus the foundation effect repository (``PlanningStorePortV1``, ``WorkflowStorePortV1``,
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

- ``workflow`` / ``run/<project_digest>/<run_key>``: the run's head document
  (record port) holding only the current record, its revision and digest and
  the sequence and rolling chain digest of the transition that produced it,
  plus the run's transition history (stream port, same key): entry 0 is the
  genesis record and every later entry is one transition with the digest of
  the record it produces. The head compare-and-swap is the commit point and
  the head's size is independent of how often the run was observed. See
  ``DurableWorkflowStoreV1`` for the chain and cursor discipline.
- ``workflow`` / ``plan/<project_digest>/<plan_fingerprint>``: plan index
  naming the run that owns the plan (at most one run per plan and project).
- ``plan`` / ``<plan_fingerprint>`` and ``plan_context`` /
  ``<provider_context_digest>``: public planning contracts, stored through
  their own ``to_dict``/``from_dict`` codecs.
- ``preparation`` / ``<preparation_digest>``: ``CanonicalPreparationV2`` bytes.
- ``execution_grant`` and ``reconciliation_grant`` / ``<slot digest>``:
  authenticated grant canonical bytes keyed by the coordinator's slot.
- ``effects`` / ``<effect_id>``: one ``EffectRecordV2`` per effect, written
  through compare-and-swap by ``DurableEffectRepositoryV1`` so the foundation
  ledger survives the process and a lost update is refused.

Composed by ``compose_reference_stores`` in this module and consumed by
``synaptic_tuner/api/v1/reference/__init__.py``. Imports no
``api/v1/persistence.py`` and no ``ProjectContext``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import json
from threading import RLock
import types
import typing

from synaptic_tuner.api.v1.planning import ProviderPlanContextV1, TrainingPlan
from synaptic_tuner.api.v1.ports import (
    DurableRecordStorePort,
    DurableStreamStorePort,
    StoragePartition,
    StoredRecordV1,
    StoredStreamEntryV1,
    StoredStreamPageV1,
)
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
    DiagnosticCode,
    FoundationError,
    canonical_bytes,
    digest_text,
    domain_digest,
    parse_canonical_object,
    safe_ref,
)
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.foundation_v2.receipts import AuthenticatedReceiptV2
from tuner.execution.foundation_v2.repository import EffectRecordV2, EffectRepositoryV2


WORKFLOW_HEAD_SCHEMA = "synaptic-reference-workflow-head/v2"
WORKFLOW_TRANSITION_SCHEMA = "synaptic-reference-workflow-transition/v1"
WORKFLOW_CHAIN_DOMAIN = "synaptic-reference-workflow-chain/v1"
WORKFLOW_PLAN_INDEX_SCHEMA = "synaptic-reference-workflow-plan-index/v1"
EFFECT_RECORD_SCHEMA = "synaptic-reference-effect/v1"
MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
HISTORY_PAGE_LIMIT = 200
APPEND_ATTEMPTS = 4

_WORKFLOW = StoragePartition.WORKFLOW.value
_PLAN = StoragePartition.PLAN.value
_PLAN_CONTEXT = StoragePartition.PLAN_CONTEXT.value
_PREPARATION = StoragePartition.PREPARATION.value
_EXECUTION_GRANT = StoragePartition.EXECUTION_GRANT.value
_RECONCILIATION_GRANT = StoragePartition.RECONCILIATION_GRANT.value
_EFFECTS = StoragePartition.EFFECTS.value


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


@dataclass(frozen=True, slots=True)
class _ValidatedRun:
    """Process-local cursor over one run: the last validated head and every revision digest.

    ``sequence`` and ``chain_digest`` name the transition-stream entry that
    produced ``record``; ``record_digests[revision]`` is the digest of every
    validated revision so ``is_descendant`` answers without another replay;
    ``head`` is the exact head bytes this state was validated from, so a read
    that finds those bytes unchanged returns without decoding them again.
    """

    sequence: int
    chain_digest: str
    record: WorkflowRecordV1
    record_digests: tuple[str, ...]
    head: bytes


def _chain_digest(previous: str | None, transition_digest: str | None, record_digest: str) -> str:
    return domain_digest(
        WORKFLOW_CHAIN_DOMAIN,
        canonical_bytes(
            {"previous": previous, "transition": transition_digest, "record": record_digest}
        ),
    )


_ENTRY_KEYS = frozenset(
    {
        "schema_version", "revision", "previous_sequence", "previous_chain_digest",
        "transition", "transition_digest", "record_digest", "chain_digest", "genesis",
    }
)
_HEAD_KEYS = frozenset(
    {"schema_version", "revision", "sequence", "chain_digest", "record_digest", "record"}
)


def _optional(document: dict[str, object], name: str, expected: type) -> object:
    value = document[name]
    if value is not None and type(value) is not expected:
        raise _integrity()
    return value


def _decode_entry(entry: StoredStreamEntryV1) -> dict[str, object]:
    """Structural decode of one transition-stream entry; chain semantics are checked by the walk."""
    document = _load(entry.canonical, WORKFLOW_TRANSITION_SCHEMA)
    if set(document) != _ENTRY_KEYS:
        raise _integrity()
    revision = document["revision"]
    if type(revision) is not int or revision < 0:
        raise _integrity()
    previous_sequence = _optional(document, "previous_sequence", int)
    previous_chain = _optional(document, "previous_chain_digest", str)
    transition = _optional(document, "transition", dict)
    transition_digest = _optional(document, "transition_digest", str)
    genesis = _optional(document, "genesis", dict)
    try:
        digest_text(document["record_digest"], "record_digest")
        digest_text(document["chain_digest"], "chain_digest")
        if previous_chain is not None:
            digest_text(previous_chain, "previous_chain_digest")
        if transition_digest is not None:
            digest_text(transition_digest, "transition_digest")
    except Exception:
        raise _integrity() from None
    is_genesis = revision == 0
    if is_genesis != (
        previous_sequence is None and previous_chain is None and transition is None
        and transition_digest is None and genesis is not None
    ):
        raise _integrity()
    if not is_genesis and (previous_sequence is None or previous_sequence >= entry.sequence):
        raise _integrity()
    return dict(document, sequence=entry.sequence)


class DurableWorkflowStoreV1:
    """``WorkflowStorePortV1`` over a head record plus an append-only transition stream.

    Layout, both under the ``workflow`` partition and keyed by
    ``run/<project_digest>/<run_key>``:

    - **Head** (record port): the current record, its revision and digest, and
      the stream ``sequence`` and rolling ``chain_digest`` of the transition
      that produced it. Its size is that of one record and is independent of
      how often the run was observed.
    - **History** (stream port): entry 0 holds the genesis record; every later
      entry holds one transition, the digest of the record it produces, its
      parent's sequence and chain digest, and its own chain digest
      ``H(previous_chain, transition_digest, record_digest)``.

    The head compare-and-swap is the commit point, exactly as before. A
    transition is appended first, then the head is swapped to point at it; an
    entry whose head swap lost a race is a dead attempt that no head ever
    references and every reader skips, because readers walk parent pointers
    back from the head rather than trusting stream order.

    Reads validate incrementally: a process-local ``_ValidatedRun`` cursor per
    run remembers the last validated head. A read whose head matches the cursor
    costs one head decode; a read that finds the head ahead of the cursor
    replays only the new entries; a cold process replays the chain once from
    genesis. Every replayed entry must re-derive its transition digest, its
    record digest and its chain digest, and the final replayed record must
    equal the head, so tampering with any retained entry or with the head is
    still detected on the next read that crosses it.
    """

    def __init__(
        self,
        records: DurableRecordStorePort,
        streams: DurableStreamStorePort,
        *,
        foundation_authenticator,
        assessment_authenticator,
        observation_authenticator,
        artifact_verifier,
    ) -> None:
        self._records = records
        self._streams = streams
        self._foundation_authenticator = foundation_authenticator
        self._assessment_authenticator = assessment_authenticator
        self._observation_authenticator = observation_authenticator
        self._artifact_verifier = artifact_verifier
        self._validated: dict[str, _ValidatedRun] = {}
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
    def _head_document(state: _ValidatedRun) -> bytes:
        record = state.record
        return _dump(
            {
                "schema_version": WORKFLOW_HEAD_SCHEMA,
                "revision": record.revision,
                "sequence": state.sequence,
                "chain_digest": state.chain_digest,
                "record_digest": record.record_digest,
                "record": encode_workflow_record(record),
            }
        )

    @staticmethod
    def _entry_document(
        *,
        revision: int,
        previous_sequence: int | None,
        previous_chain_digest: str | None,
        transition: CoordinatorTransitionV1 | None,
        transition_digest: str | None,
        record_digest: str,
        chain_digest: str,
        genesis: WorkflowRecordV1 | None,
    ) -> bytes:
        return _dump(
            {
                "schema_version": WORKFLOW_TRANSITION_SCHEMA,
                "revision": revision,
                "previous_sequence": previous_sequence,
                "previous_chain_digest": previous_chain_digest,
                "transition": None if transition is None else encode_transition(transition),
                "transition_digest": transition_digest,
                "record_digest": record_digest,
                "chain_digest": chain_digest,
                "genesis": None if genesis is None else encode_workflow_record(genesis),
            }
        )

    def _load_head(self, raw: bytes, key: str) -> tuple[WorkflowRecordV1, int, str]:
        document = _load(raw, WORKFLOW_HEAD_SCHEMA)
        if set(document) != _HEAD_KEYS:
            raise _integrity()
        revision = document["revision"]
        sequence = document["sequence"]
        chain_digest = document["chain_digest"]
        if (
            type(revision) is not int
            or type(sequence) is not int
            or not 0 <= revision <= sequence
            or type(chain_digest) is not str
        ):
            raise _integrity()
        try:
            digest_text(chain_digest, "chain_digest")
            record = decode_workflow_record(document["record"])
        except Exception:
            raise _integrity() from None
        if (
            record.revision != revision
            or record.record_digest != document["record_digest"]
            or self._head_key(record.run) != key
        ):
            raise _integrity()
        return record, sequence, chain_digest

    def _replay(self, current: WorkflowRecordV1, transition: CoordinatorTransitionV1) -> WorkflowRecordV1:
        return replay_transition(
            current,
            transition,
            foundation_authenticator=self._foundation_authenticator,
            assessment_authenticator=self._assessment_authenticator,
            observation_authenticator=self._observation_authenticator,
            artifact_verifier=self._artifact_verifier,
        )

    # -- transition stream -------------------------------------------------

    def _read_page(self, key: str, after_sequence: int | None) -> StoredStreamPageV1:
        try:
            page = self._streams.read_page(
                partition=_WORKFLOW,
                stream_key=key,
                after_sequence=after_sequence,
                limit=HISTORY_PAGE_LIMIT,
            )
        except Exception:
            raise _integrity() from None
        if type(page) is not StoredStreamPageV1:
            raise _integrity()
        return page

    def _entries(self, key: str, *, after_sequence: int | None, until_sequence: int) -> dict[int, dict[str, object]]:
        """Decoded entries after ``after_sequence`` up to and including ``until_sequence``."""
        found: dict[int, dict[str, object]] = {}
        cursor = after_sequence
        while True:
            page = self._read_page(key, cursor)
            for entry in page.entries:
                if entry.sequence > until_sequence:
                    return found
                found[entry.sequence] = _decode_entry(entry)
            if not page.truncated or page.next_cursor is None or page.next_cursor >= until_sequence:
                return found
            cursor = page.next_cursor

    def _tail_sequence(self, key: str, *, after_sequence: int) -> int | None:
        tail = None
        cursor = after_sequence
        while True:
            page = self._read_page(key, cursor)
            if page.entries:
                tail = page.entries[-1].sequence
            if not page.truncated or page.next_cursor is None:
                return tail
            cursor = page.next_cursor

    def _append(self, key: str, *, after_sequence: int, canonical: bytes) -> int | None:
        """Append at the next free sequence; ``None`` when the stream keeps racing ahead."""
        sequence = after_sequence + 1
        for _ in range(APPEND_ATTEMPTS):
            try:
                appended = self._streams.append(
                    partition=_WORKFLOW, stream_key=key, sequence=sequence, canonical=canonical
                )
            except Exception:
                raise _integrity() from None
            if appended is True:
                return sequence
            if appended is not False:
                raise _integrity()
            tail = self._tail_sequence(key, after_sequence=sequence - 1)
            if tail is None or tail < sequence:
                raise _integrity()
            sequence = tail + 1
        return None

    # -- validation ----------------------------------------------------------

    def _validated_state(self, key: str, stored: StoredRecordV1) -> _ValidatedRun:
        """Prove the head continues the validated chain, replaying only what is new."""
        cached = self._validated.get(key)
        if cached is not None and cached.head == stored.canonical:
            return cached
        head, sequence, chain_digest = self._load_head(stored.canonical, key)
        if cached is not None:
            if cached.sequence > sequence:
                raise _integrity()
            if cached.sequence == sequence:
                if (
                    cached.chain_digest != chain_digest
                    or cached.record.record_digest != head.record_digest
                    or cached.record != head
                ):
                    raise _integrity()
                cached = _ValidatedRun(
                    cached.sequence, cached.chain_digest, cached.record, cached.record_digests, stored.canonical
                )
                self._validated[key] = cached
                return cached
        start_sequence = None if cached is None else cached.sequence
        entries = self._entries(key, after_sequence=start_sequence, until_sequence=sequence)
        path: list[dict[str, object]] = []
        cursor: int | None = sequence
        while cursor != start_sequence:
            if cursor is None or len(path) > len(entries):
                raise _integrity()
            entry = entries.get(cursor)
            if entry is None:
                raise _integrity()
            path.append(entry)
            cursor = entry["previous_sequence"]
        previous_record = None if cached is None else cached.record
        previous_chain = None if cached is None else cached.chain_digest
        previous_sequence = start_sequence
        digests = [] if cached is None else list(cached.record_digests)
        try:
            for entry in reversed(path):
                if (
                    entry["previous_sequence"] != previous_sequence
                    or entry["previous_chain_digest"] != previous_chain
                ):
                    raise _integrity()
                if previous_record is None:
                    if entry["revision"] != 0:
                        raise _integrity()
                    record = decode_workflow_record(entry["genesis"])
                    if (
                        record.phase is not WorkflowPhaseV1.PLANNED
                        or record.revision != 0
                        or self._head_key(record.run) != key
                    ):
                        raise _integrity()
                    transition_digest = None
                else:
                    if entry["revision"] != previous_record.revision + 1 or entry["transition"] is None:
                        raise _integrity()
                    transition = decode_transition(entry["transition"])
                    _, transition_digest = revalidate_transition(transition)
                    if entry["transition_digest"] != transition_digest:
                        raise _integrity()
                    record = self._replay(previous_record, transition)
                    if (
                        record.revision != previous_record.revision + 1
                        or record.run != previous_record.run
                        or record.plan_fingerprint != previous_record.plan_fingerprint
                    ):
                        raise _integrity()
                if entry["record_digest"] != record.record_digest:
                    raise _integrity()
                chain = _chain_digest(previous_chain, transition_digest, record.record_digest)
                if entry["chain_digest"] != chain:
                    raise _integrity()
                digests.append(record.record_digest)
                previous_record, previous_chain, previous_sequence = record, chain, entry["sequence"]
        except CoordinatorStoreError:
            raise _integrity() from None
        except Exception:
            raise _integrity() from None
        if (
            previous_record is None
            or previous_sequence != sequence
            or previous_chain != chain_digest
            or previous_record.record_digest != head.record_digest
            or previous_record != head
            or len(digests) != head.revision + 1
        ):
            raise _integrity()
        state = _ValidatedRun(sequence, chain_digest, head, tuple(digests), stored.canonical)
        self._validated[key] = state
        return state

    def _retained(self, stored: StoredRecordV1, key: str) -> WorkflowRecordV1:
        retained = self._validated_state(key, stored).record
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
            chain = _chain_digest(None, None, candidate.record_digest)
            genesis = self._entry_document(
                revision=0,
                previous_sequence=None,
                previous_chain_digest=None,
                transition=None,
                transition_digest=None,
                record_digest=candidate.record_digest,
                chain_digest=chain,
                genesis=candidate,
            )
            try:
                appended = self._streams.append(
                    partition=_WORKFLOW, stream_key=key, sequence=0, canonical=genesis
                )
            except Exception:
                raise _integrity() from None
            if appended is not True:
                first = self._read_page(key, None).entries
                if not first or first[0].sequence != 0:
                    raise _integrity()
                if first[0].canonical != genesis:
                    raise _closed(CoordinatorStoreCode.CONFLICT)
            state = _ValidatedRun(0, chain, candidate, (candidate.record_digest,), b"")
            head = self._head_document(state)
            try:
                created = self._records.create(partition=_WORKFLOW, key=key, canonical=head)
            except Exception:
                raise _integrity() from None
            if created is True:
                self._validated[key] = _ValidatedRun(0, chain, candidate, (candidate.record_digest,), head)
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
        with self._lock:
            stored = _read(self._records, _WORKFLOW, key)
            if stored is None:
                return None
            retained = self._retained(stored, key)
        if retained.run != run:
            raise _integrity()
        return retained

    def get_by_plan(self, project_ref: str, plan_fingerprint: str) -> WorkflowRecordV1 | None:
        plan_key = self._plan_key(project_ref, plan_fingerprint)
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            stored = _read(self._records, _WORKFLOW, key)
            if stored is None:
                return False
            state = self._validated_state(key, stored)
        if ancestor.revision >= descendant.revision or descendant.revision > state.record.revision:
            return False
        return bool(
            state.record_digests[ancestor.revision] == ancestor.record_digest
            and state.record_digests[descendant.revision] == descendant.record_digest
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
            state = self._validated_state(key, stored)
            retained = state.record
            if retained.revision != expected.revision:
                return False
            if retained.record_digest != expected.record_digest or retained != expected:
                raise _closed(CoordinatorStoreCode.CONFLICT)
            replayed = self._replay(retained, transition)
            if replayed.revision != retained.revision + 1 or replayed != replacement:
                raise _closed(CoordinatorStoreCode.TRANSITION_INVALID)
            transition, transition_digest = revalidate_transition(transition)
            chain = _chain_digest(state.chain_digest, transition_digest, replacement.record_digest)
            entry = self._entry_document(
                revision=replacement.revision,
                previous_sequence=state.sequence,
                previous_chain_digest=state.chain_digest,
                transition=transition,
                transition_digest=transition_digest,
                record_digest=replacement.record_digest,
                chain_digest=chain,
                genesis=None,
            )
            sequence = self._append(key, after_sequence=state.sequence, canonical=entry)
            if sequence is None:
                return False
            advanced = _ValidatedRun(
                sequence, chain, replacement, state.record_digests + (replacement.record_digest,), b""
            )
            head = self._head_document(advanced)
            try:
                swapped = self._records.compare_and_swap(
                    partition=_WORKFLOW, key=key, expected_revision=stored.revision, canonical=head
                )
            except Exception:
                raise _integrity() from None
            if type(swapped) is not bool:
                raise _integrity()
            if swapped:
                self._validated[key] = _ValidatedRun(
                    sequence, chain, replacement, advanced.record_digests, head
                )
            return swapped


# --------------------------------------------------------------------------
# Effect ledger over the ``effects`` partition
# --------------------------------------------------------------------------


def encode_effect_record(record: EffectRecordV2, effect_id: str) -> bytes:
    return _dump(
        {
            "schema_version": EFFECT_RECORD_SCHEMA,
            "effect_id": effect_id,
            "record": _encode(record, EffectRecordV2),
        }
    )


def decode_effect_record(raw: bytes, effect_id: str) -> EffectRecordV2:
    """Decode a stored effect record; every failure is one closed foundation code."""
    try:
        document = _load(raw, EFFECT_RECORD_SCHEMA)
        if set(document) != {"schema_version", "effect_id", "record"} or document["effect_id"] != effect_id:
            raise _integrity()
        record = _decode(document["record"], EffectRecordV2)
        if parse_exact_command(record.command_bytes).operation.effect.effect_id != effect_id:
            raise _integrity()
    except Exception:
        raise FoundationError(DiagnosticCode.AUTHORITY_INVALID) from None
    return record


class _DurableEffectRecords:
    """The ``effect_id -> EffectRecordV2`` mapping ``EffectRepositoryV2`` writes through.

    Every write is bound to the store revision of the record this instance
    last read for that key: a first write is ``create``, a later one is
    ``compare_and_swap``. A refused write means another writer moved the
    record first and surfaces as ``effect_conflict`` instead of a lost update.
    The repository serialises its own read-then-write under its lock. A read
    whose stored bytes equal the bytes this instance last decoded or wrote for
    the key returns that record without decoding it again; the repository
    still re-authenticates every record it reads.
    """

    __slots__ = ("_records", "_revisions", "_cached")

    def __init__(self, records: DurableRecordStorePort) -> None:
        self._records = records
        self._revisions: dict[str, int] = {}
        self._cached: dict[str, tuple[bytes, EffectRecordV2]] = {}

    @staticmethod
    def _key(effect_id: object) -> str:
        try:
            return safe_ref(effect_id, "effect_id")
        except Exception:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH) from None

    def get(self, effect_id: object) -> EffectRecordV2 | None:
        key = self._key(effect_id)
        try:
            stored = self._records.read(partition=_EFFECTS, key=key)
        except Exception:
            raise FoundationError(DiagnosticCode.AUTHORITY_INVALID) from None
        if stored is None:
            self._revisions.pop(key, None)
            self._cached.pop(key, None)
            return None
        if type(stored) is not StoredRecordV1 or stored.key != key:
            raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        cached = self._cached.get(key)
        if cached is not None and cached[0] == stored.canonical:
            record = cached[1]
        else:
            record = decode_effect_record(stored.canonical, key)
            self._cached[key] = (stored.canonical, record)
        self._revisions[key] = stored.revision
        return record

    def __getitem__(self, effect_id: object) -> EffectRecordV2:
        record = self.get(effect_id)
        if record is None:
            raise KeyError(effect_id)
        return record

    def __setitem__(self, effect_id: object, record: EffectRecordV2) -> None:
        key = self._key(effect_id)
        if type(record) is not EffectRecordV2:
            raise FoundationError(DiagnosticCode.AUTHORITY_INVALID)
        try:
            canonical = encode_effect_record(record, key)
        except Exception:
            raise FoundationError(DiagnosticCode.AUTHORITY_INVALID) from None
        revision = self._revisions.get(key)
        try:
            if revision is None:
                admitted = self._records.create(partition=_EFFECTS, key=key, canonical=canonical)
                advanced = 1
            else:
                admitted = self._records.compare_and_swap(
                    partition=_EFFECTS, key=key, expected_revision=revision, canonical=canonical
                )
                advanced = revision + 1
        except Exception:
            raise FoundationError(DiagnosticCode.AUTHORITY_INVALID) from None
        if admitted is not True:
            self._revisions.pop(key, None)
            self._cached.pop(key, None)
            raise FoundationError(DiagnosticCode.EFFECT_CONFLICT)
        self._revisions[key] = advanced
        self._cached[key] = (canonical, record)


class DurableEffectRepositoryV1(EffectRepositoryV2):
    """``EffectRepositoryV2`` persisted through the host record store's ``effects`` partition.

    Same reduction logic as ``InMemoryEffectRepositoryV2``; only the record
    mapping differs, so a host that recomposes over the same store finds every
    effect, its receipts and its reconciliation claims exactly where the
    previous process left them.
    """

    def __init__(
        self,
        records: DurableRecordStorePort,
        *,
        receipt_authority,
        invalid_evidence_authority,
        recovery_verifier,
        finality_verifier,
        grant_authority,
    ) -> None:
        super().__init__(
            receipt_authority,
            invalid_evidence_authority,
            recovery_verifier,
            finality_verifier,
            grant_authority,
            _DurableEffectRecords(records),
        )


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
    "DurableEffectRepositoryV1",
    "DurableExecutionGrantStoreV1",
    "DurablePlanningStoreV1",
    "DurablePreparationStoreV1",
    "DurableReconciliationGrantStoreV1",
    "DurableWorkflowStoreV1",
    "EFFECT_RECORD_SCHEMA",
    "MAX_DOCUMENT_BYTES",
    "WORKFLOW_HEAD_SCHEMA",
    "WORKFLOW_PLAN_INDEX_SCHEMA",
    "WORKFLOW_TRANSITION_SCHEMA",
    "decode_effect_record",
    "decode_execution_grant",
    "decode_reconciliation_grant",
    "decode_transition",
    "decode_workflow_record",
    "encode_effect_record",
    "encode_transition",
    "encode_workflow_record",
]
