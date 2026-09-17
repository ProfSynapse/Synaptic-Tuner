"""Data reference composition: ``DataOperations`` over SynthChat generation and improvement.

Location: ``synaptic_tuner/api/v1/reference/data.py``.

``ReferenceDataOperationsV1`` implements ``DataOperations``
(``api/v1/data_facade.py``: plan, preflight, start, show, cancel, list,
datasets, validate, observations) over the public host ports. One run produces
one dataset artifact; SynthChat's ``SynthChatGenerator`` writes the rows and
``ImprovementEngine`` improves them, both through ``StreamingResultWriter``,
whose data file is homogeneous JSONL and whose metadata lives in a
``.meta.json`` sidecar. The reference hashes both files into the
``dataset_jsonl`` and ``dataset_metadata`` ``VerifiedArtifact`` values.

Durable layout, all under the ``data`` partition and keyed by
``run/<project_digest>/<run_id>``:

- **Head** (record port): the current ``synaptic-dataset-result/v1`` record,
  the plan it executes, the event count and the rolling chain digest. Its
  size is independent of how many rows the run wrote.
- **History** (stream port, same key): one entry per state transition
  (``planned``, ``running``, optionally ``cancel_requested``, then one
  terminal state), so a run never exceeds four durable events; the budget the
  architecture grants the family is twelve.
- ``plan/<project_digest>/<plan_fingerprint>`` names the run a plan started,
  which makes ``start`` idempotent for an identical plan.

Per-row and per-scenario progress goes to the ``observation`` partition as
``synaptic-observation/v1`` records under ``data/<project_digest>/<run_id>``.

``start`` runs the whole generation synchronously in the caller's thread and
returns once the run is terminal. Cancellation is cooperative: ``cancel``
records ``cancel_requested`` on the head, and the loop checks the head at
every row and scenario boundary, finishing as ``cancelled`` with the artifact
produced so far. A backend failure after some rows were written finishes as
``partially_succeeded``, also with the artifact.

Local backends only in this slice: an unknown backend is refused with
``backend_unavailable`` and a metered one (openrouter, openai, ...) with
``backend_unmetered`` until the usage-metering slice lands. A local run claims
no ``spend`` effect, so nothing is written to the ``effects`` partition; the
dataset write itself is evidenced by the artifact digests, not ledgered.

The host supplies the SynthChat config, scenario and rubric directories and an
LLM client factory through ``ReferenceDataPortsV1``; no path is derived from a
project context and no environment variable or ``.env`` file is read here.
SynthChat is imported lazily inside the executing methods so composing a host
does not load it. No exception text or traceback ever reaches a record: the
only free text a record carries is the redacted cancel reason.

Consumed by ``synaptic_tuner/api/v1/reference/composition.py``.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Protocol

from synaptic_tuner.api.v1.data_facade import (
    DATA_PLAN_SCHEMA_VERSION,
    DATA_RESULT_SCHEMA_VERSION,
    DATASET_JSONL_ROLE,
    DATASET_METADATA_ROLE,
    DataListRequest,
    DataMode,
    DataOperationCode,
    DataOperationError,
    DataOutcome,
    DataPage,
    DataPlan,
    DataPreflight,
    DataRequest,
    DataResult,
    DataRunRef,
    DataRunState,
    DataScenarioOutcome,
    DataScenarioTarget,
    DataStart,
    DatasetDescriptor,
    DatasetListRequest,
    DatasetPage,
    DatasetValidateRequest,
    ValidationFinding,
    ValidationFindingCode,
    ValidationReport,
)
from synaptic_tuner.api.v1.observations import (
    OBSERVATION_SCHEMA_VERSION,
    DataRowWrittenPayloadV1,
    DataScenarioCompletedPayloadV1,
    DataStageGateEvaluatedPayloadV1,
    ObservationFamily,
    ObservationKind,
    ObservationPage,
    ObservationRecordV1,
    ObservationStreamRef,
    ObservationsRequest,
)
from synaptic_tuner.api.v1.ports import (
    ClockPort,
    DurableRecordStorePort,
    DurableStreamStorePort,
    StoragePartition,
    StoredRecordV1,
)
from synaptic_tuner.api.v1.results import VerifiedArtifact
from synaptic_tuner.api.v1.training_facade import AuthorizationRequirement
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest, parse_canonical_object, safe_ref
from tuner.execution.redaction import redact


DATA_HEAD_SCHEMA = "synaptic-reference-data-head/v1"
DATA_EVENT_SCHEMA = "synaptic-reference-data-event/v1"
DATA_PLAN_INDEX_SCHEMA = "synaptic-reference-data-plan-index/v1"
DATA_START_OPERATION = "data.start"
DATA_DURABLE_EVENT_BUDGET = 12
DATA_PREFLIGHT_SECONDS = 900

LOCAL_BACKENDS = frozenset({"lmstudio", "ollama", "unsloth"})
METERED_BACKENDS = frozenset({"openrouter", "openai", "openai_responses", "anthropic"})

DIAGNOSTIC_BACKEND_UNAVAILABLE = "backend_unavailable"
DIAGNOSTIC_GENERATION_FAILED = "generation_failed"
DIAGNOSTIC_OUTPUT_FAILED = "output_failed"
DIAGNOSTIC_CANCELLED = "cancelled"

_DATA = StoragePartition.DATA.value
_OBSERVATION = StoragePartition.OBSERVATION.value
_CURSOR_PREFIX = "v1."
_LOGGER = logging.getLogger(__name__)


class LLMClientFactoryPort(Protocol):
    """Host-supplied construction of a ``shared.llm`` style client for a local backend.

    The reference never reads hosts, ports or keys from the environment; the
    host builds the client from configuration it owns.
    """

    def create(self, *, backend: str, model: str) -> object: ...


@dataclass(frozen=True, slots=True)
class ReferenceDataPortsV1:
    """SynthChat locations and the LLM client factory a reference data host supplies."""

    config_dir: Path
    scenarios_dir: Path
    rubrics_dir: Path
    llm_clients: LLMClientFactoryPort

    def __post_init__(self) -> None:
        for name in ("config_dir", "scenarios_dir", "rubrics_dir"):
            value = getattr(self, name)
            if type(value) is not Path and not isinstance(value, Path):
                raise TypeError(f"{name} must be a Path")
            if not value.is_absolute():
                raise ValueError(f"{name} must be an absolute path")
        if not callable(getattr(type(self.llm_clients), "create", None)):
            raise TypeError("llm_clients must expose create()")


# --- helpers ----------------------------------------------------------------------


def _closed(code: DataOperationCode) -> DataOperationError:
    return DataOperationError(code)


def _integrity() -> DataOperationError:
    return DataOperationError(DataOperationCode.INTEGRITY_ERROR)


def _project_digest(project_ref: str) -> str:
    try:
        safe_ref(project_ref, "project_ref")
    except Exception:
        raise _closed(DataOperationCode.MANIFEST_INVALID) from None
    return domain_digest("synaptic-reference-project-key/v1", project_ref.encode("utf-8"))


def data_run_ref(plan: DataPlan) -> DataRunRef:
    """The run identity a plan starts; derivation-only, so an identical plan names one run."""
    if type(plan) is not DataPlan:
        raise TypeError("plan must be exact DataPlan")
    digest = domain_digest("synaptic-reference-data-run/v1", plan.plan_fingerprint.encode("ascii"))
    return DataRunRef(f"data-{digest[:32]}", plan.request.project_ref)


def _run_key(run: DataRunRef) -> str:
    return f"run/{_project_digest(run.project_ref)}/{run.run_id}"


def _plan_key(plan: DataPlan) -> str:
    return f"plan/{_project_digest(plan.request.project_ref)}/{plan.plan_fingerprint}"


def _observation_key(project_ref: str, run_id: str) -> str:
    return f"data/{_project_digest(project_ref)}/{run_id}"


def _encode_cursor(value: str) -> str:
    return _CURSOR_PREFIX + base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str | None) -> str | None:
    if cursor is None:
        return None
    if type(cursor) is not str or not cursor.startswith(_CURSOR_PREFIX):
        raise _closed(DataOperationCode.CURSOR_INVALID)
    body = cursor[len(_CURSOR_PREFIX):]
    try:
        raw = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
        decoded = raw.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        raise _closed(DataOperationCode.CURSOR_INVALID) from None
    if not decoded:
        raise _closed(DataOperationCode.CURSOR_INVALID)
    return decoded


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise _closed(DataOperationCode.MANIFEST_INVALID)
    return path


def _sidecar_path(output: Path) -> Path:
    """SynthChat owns the sidecar convention; ``result_writer`` is stdlib-only."""
    from SynthChat.result_writer import metadata_path

    return metadata_path(output)


def _verified_artifact(role: str, path: Path) -> VerifiedArtifact:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return VerifiedArtifact(role, digest.hexdigest(), size)


def _dump(document: dict) -> bytes:
    return canonical_bytes(document)


def _load(raw: bytes, schema_version: str) -> dict:
    try:
        document = parse_canonical_object(raw, name="data record")
    except Exception:
        raise _integrity() from None
    if type(document) is not dict or document.get("schema_version") != schema_version:
        raise _integrity()
    return document


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


def _check_backend(backend: str) -> None:
    if backend in METERED_BACKENDS:
        raise _closed(DataOperationCode.BACKEND_UNMETERED)
    if backend not in LOCAL_BACKENDS:
        raise _closed(DataOperationCode.BACKEND_UNAVAILABLE)


def _read_dataset_rows(path: Path) -> list[dict]:
    """Every non-blank line as an object; a malformed dataset is not an improvable manifest."""
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                row = json.loads(text)
                if type(row) is not dict or "_meta" in row:
                    raise ValueError("row")
                rows.append(row)
    except (OSError, ValueError):
        raise _closed(DataOperationCode.MANIFEST_INVALID) from None
    return rows


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


@dataclass(frozen=True, slots=True)
class _Head:
    plan: DataPlan
    result: DataResult
    revision: int
    sequence: int
    chain_digest: str
    stored_revision: int


@dataclass
class _WrittenRow:
    example: dict


class _Cancelled(Exception):
    """Internal signal: the head asked the loop to stop at a boundary."""


# --- operations ---------------------------------------------------------------------


class ReferenceDataOperationsV1:
    """``DataOperations`` over the host record and stream stores and SynthChat."""

    def __init__(
        self,
        *,
        records: DurableRecordStorePort,
        streams: DurableStreamStorePort,
        clock: ClockPort,
        ports: ReferenceDataPortsV1,
    ) -> None:
        if type(ports) is not ReferenceDataPortsV1:
            raise TypeError("exact ReferenceDataPortsV1 required")
        self._records = records
        self._streams = streams
        self._clock = clock
        self._ports = ports
        self._lock = RLock()
        self._cancel_flags: set[str] = set()

    # -- plan and preflight ------------------------------------------------------

    def _scenario_definitions(self, request: DataRequest) -> dict[str, object]:
        from SynthChat.generator import ScenarioLoader

        try:
            loader = ScenarioLoader(self._ports.scenarios_dir)
        except Exception:
            raise _closed(DataOperationCode.SCENARIO_INVALID) from None
        definitions: dict[str, object] = {}
        for target in request.scenarios:
            scenario = loader.get_scenario(target.scenario_ref)
            if not isinstance(scenario, dict):
                raise _closed(DataOperationCode.SCENARIO_INVALID)
            definitions[target.scenario_ref] = _json_safe(scenario)
        return definitions

    def _rubric_check(self, request: DataRequest) -> None:
        for ref in request.rubric_refs:
            if not (self._ports.rubrics_dir / f"{ref}.yaml").is_file():
                raise _closed(DataOperationCode.MANIFEST_INVALID)

    def _resolve(self, request: DataRequest) -> tuple[tuple[DataScenarioTarget, ...], int, str]:
        """Scenario targets, row total and source digest for a request."""
        _absolute_path(request.output_ref)
        if request.mode is DataMode.GENERATE:
            definitions = self._scenario_definitions(request)
            digest = domain_digest(
                "synaptic-reference-data-source/v1",
                json.dumps(definitions, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
            )
            return request.scenarios, sum(item.rows for item in request.scenarios), digest
        self._rubric_check(request)
        input_path = _absolute_path(request.input_dataset_ref or "")
        if not input_path.is_file():
            raise _closed(DataOperationCode.DATASET_MISSING)
        rows = _read_dataset_rows(input_path)
        if not rows:
            raise _closed(DataOperationCode.MANIFEST_INVALID)
        digest = domain_digest(
            "synaptic-reference-data-source/v1",
            canonical_bytes({
                "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "rubric_refs": list(request.rubric_refs),
            }),
        )
        target = DataScenarioTarget(request.input_dataset_ref, len(rows))  # type: ignore[arg-type]
        return (target,), len(rows), digest

    def plan(self, request: DataRequest) -> DataPlan:
        if type(request) is not DataRequest:
            raise TypeError("request must be exact DataRequest")
        scenarios, rows, digest = self._resolve(request)
        return DataPlan(DATA_PLAN_SCHEMA_VERSION, request, scenarios, rows, digest)

    def _verify_plan(self, plan: DataPlan) -> None:
        if type(plan) is not DataPlan:
            raise TypeError("plan must be exact DataPlan")
        scenarios, rows, digest = self._resolve(plan.request)
        if (scenarios, rows, digest) != (plan.scenarios, plan.rows_requested, plan.source_digest):
            raise _closed(DataOperationCode.MANIFEST_INVALID)

    def preflight(self, plan: DataPlan) -> DataPreflight:
        self._verify_plan(plan)
        checked_at = self._clock.now()
        expires_at = _plus_seconds(checked_at, DATA_PREFLIGHT_SECONDS)
        codes: list[str] = []
        backend = plan.request.backend
        if backend in METERED_BACKENDS:
            codes.append(DataOperationCode.BACKEND_UNMETERED.value)
        elif backend not in LOCAL_BACKENDS:
            codes.append(DataOperationCode.BACKEND_UNAVAILABLE.value)
        if self._existing_run(plan) is None and self._output_occupied(plan):
            codes.append(DataOperationCode.OUTPUT_CONFLICT.value)
        return DataPreflight(
            plan.plan_fingerprint,
            not codes,
            checked_at,
            expires_at,
            (AuthorizationRequirement(DATA_START_OPERATION, False),),
            tuple(codes),
        )

    @staticmethod
    def _output_occupied(plan: DataPlan) -> bool:
        output = _absolute_path(plan.request.output_ref)
        sidecar = _sidecar_path(output)
        return (output.exists() and output.stat().st_size > 0) or sidecar.exists()

    # -- durable head and history --------------------------------------------------

    def _head_document(self, head: _Head) -> bytes:
        return _dump({
            "schema_version": DATA_HEAD_SCHEMA,
            "revision": head.revision,
            "sequence": head.sequence,
            "chain_digest": head.chain_digest,
            "plan": head.plan.to_dict(),
            "result": head.result.to_dict(),
        })

    def _decode_head(self, stored: StoredRecordV1) -> _Head:
        document = _load(stored.canonical, DATA_HEAD_SCHEMA)
        if set(document) != {"schema_version", "revision", "sequence", "chain_digest", "plan", "result"}:
            raise _integrity()
        try:
            plan = DataPlan.from_dict(document["plan"])
            result = DataResult.from_dict(document["result"])
            revision = document["revision"]
            sequence = document["sequence"]
            chain = document["chain_digest"]
            if type(revision) is not int or type(sequence) is not int or type(chain) is not str:
                raise ValueError("head")
            if revision != sequence + 1 or revision > DATA_DURABLE_EVENT_BUDGET:
                raise ValueError("head")
            if result.run != data_run_ref(plan):
                raise ValueError("head")
        except Exception:
            raise _integrity() from None
        return _Head(plan, result, revision, sequence, chain, stored.revision)

    def _read_head(self, run: DataRunRef) -> _Head | None:
        stored = _read(self._records, _DATA, _run_key(run))
        return None if stored is None else self._decode_head(stored)

    def _require_head(self, run: DataRunRef) -> _Head:
        head = self._read_head(run)
        if head is None:
            raise _closed(DataOperationCode.RUN_MISSING)
        return head

    def _existing_run(self, plan: DataPlan) -> DataRunRef | None:
        stored = _read(self._records, _DATA, _plan_key(plan))
        if stored is None:
            return None
        document = _load(stored.canonical, DATA_PLAN_INDEX_SCHEMA)
        if set(document) != {"schema_version", "run"}:
            raise _integrity()
        try:
            return DataRunRef.from_dict(document["run"])
        except Exception:
            raise _integrity() from None

    def _event_document(self, *, revision: int, occurred_at: str, result: DataResult, previous_chain: str | None, note: str | None) -> tuple[bytes, str]:
        result_digest = domain_digest(DATA_RESULT_SCHEMA_VERSION, _dump(result.to_dict()))
        chain = domain_digest(
            "synaptic-reference-data-chain/v1",
            canonical_bytes({"previous": previous_chain, "record": result_digest, "revision": revision}),
        )
        document = _dump({
            "schema_version": DATA_EVENT_SCHEMA,
            "revision": revision,
            "occurred_at": occurred_at,
            "state": result.state.value,
            "diagnostic_code": result.diagnostic_code,
            "note": note,
            "previous_chain_digest": previous_chain,
            "record_digest": result_digest,
            "chain_digest": chain,
        })
        return document, chain

    def _append_history(self, run: DataRunRef, sequence: int, document: bytes) -> None:
        try:
            admitted = self._streams.append(partition=_DATA, stream_key=_run_key(run), sequence=sequence, canonical=document)
        except Exception:
            raise _integrity() from None
        if admitted is not True:
            raise _closed(DataOperationCode.STATE_CONFLICT)

    def _genesis(self, plan: DataPlan, result: DataResult) -> _Head:
        run = result.run
        document, chain = self._event_document(
            revision=0, occurred_at=self._clock.now(), result=result, previous_chain=None, note=None,
        )
        head = _Head(plan, result, 0 + 1, 0, chain, 1)
        try:
            created = self._records.create(partition=_DATA, key=_run_key(run), canonical=self._head_document(head))
        except Exception:
            raise _integrity() from None
        if created is not True:
            raise _closed(DataOperationCode.STATE_CONFLICT)
        self._append_history(run, 0, document)
        index = _dump({"schema_version": DATA_PLAN_INDEX_SCHEMA, "run": run.to_dict()})
        try:
            self._records.put_if_absent(partition=_DATA, key=_plan_key(plan), canonical=index)
        except Exception:
            raise _integrity() from None
        return head

    def _transition(self, run: DataRunRef, mutate, *, note: str | None = None) -> _Head:
        """Append one history entry and swap the head to it; ``mutate`` maps result -> result."""
        with self._lock:
            head = self._require_head(run)
            result = mutate(head.result)
            if result == head.result:
                return head
            if head.revision >= DATA_DURABLE_EVENT_BUDGET:
                raise _integrity()
            sequence = head.sequence + 1
            document, chain = self._event_document(
                revision=head.revision, occurred_at=self._clock.now(), result=result,
                previous_chain=head.chain_digest, note=note,
            )
            self._append_history(run, sequence, document)
            updated = _Head(head.plan, result, head.revision + 1, sequence, chain, head.stored_revision + 1)
            try:
                swapped = self._records.compare_and_swap(
                    partition=_DATA, key=_run_key(run), expected_revision=head.stored_revision,
                    canonical=self._head_document(updated),
                )
            except Exception:
                raise _integrity() from None
            if swapped is not True:
                raise _closed(DataOperationCode.STATE_CONFLICT)
            return updated

    # -- observations ---------------------------------------------------------------

    def _observe(self, run: DataRunRef, sequence: int, kind: ObservationKind, payload) -> None:
        record = ObservationRecordV1(
            OBSERVATION_SCHEMA_VERSION,
            ObservationStreamRef(ObservationFamily.DATA, run.project_ref, run.run_id),
            sequence,
            self._clock.now(),
            kind,
            payload,
        )
        try:
            self._streams.append(
                partition=_OBSERVATION,
                stream_key=_observation_key(run.project_ref, run.run_id),
                sequence=sequence,
                canonical=_dump(record.to_dict()),
            )
        except Exception:
            raise _integrity() from None

    def observations(self, request: ObservationsRequest) -> ObservationPage:
        if type(request) is not ObservationsRequest:
            raise TypeError("request must be exact ObservationsRequest")
        stream = request.stream
        if stream.family is not ObservationFamily.DATA:
            raise _closed(DataOperationCode.MANIFEST_INVALID)
        try:
            page = self._streams.read_page(
                partition=_OBSERVATION,
                stream_key=_observation_key(stream.project_ref, stream.entity_id),
                after_sequence=request.after_sequence,
                limit=request.limit,
            )
        except Exception:
            raise _integrity() from None
        records = []
        for entry in page.entries:
            try:
                record = ObservationRecordV1.from_dict(parse_canonical_object(entry.canonical, name="observation"))
            except Exception:
                raise _integrity() from None
            if record.stream != stream or record.sequence != entry.sequence:
                raise _integrity()
            records.append(record)
        return ObservationPage(request, tuple(records), page.next_cursor, page.truncated)

    # -- start and the run loop -------------------------------------------------------

    def _initial_result(self, plan: DataPlan, run: DataRunRef, state: DataRunState) -> DataResult:
        return DataResult(
            DATA_RESULT_SCHEMA_VERSION, run, state, plan.request.mode, plan.request.backend,
            plan.request.model, plan.rows_requested, 0,
            tuple(DataScenarioOutcome(item.scenario_ref, item.rows, 0) for item in plan.scenarios),
        )

    def start(self, plan: DataPlan) -> DataStart:
        self._verify_plan(plan)
        _check_backend(plan.request.backend)
        existing = self._existing_run(plan)
        if existing is not None:
            self._require_head(existing)
            return DataStart(existing, True)
        if self._output_occupied(plan):
            raise _closed(DataOperationCode.OUTPUT_CONFLICT)
        try:
            client = self._ports.llm_clients.create(backend=plan.request.backend, model=plan.request.model)
        except Exception:
            raise _closed(DataOperationCode.BACKEND_UNAVAILABLE) from None
        run = data_run_ref(plan)
        self._genesis(plan, self._initial_result(plan, run, DataRunState.PLANNED))
        self._transition(run, lambda result: _with_state(result, DataRunState.RUNNING))
        self._execute(plan, run, client)
        return DataStart(run, True)

    def _cancel_requested(self, run: DataRunRef) -> bool:
        with self._lock:
            if run.run_id in self._cancel_flags:
                return True
        return self._require_head(run).result.state is DataRunState.CANCEL_REQUESTED

    def _execute(self, plan: DataPlan, run: DataRunRef, client) -> None:
        from SynthChat.result_writer import StreamingResultWriter

        request = plan.request
        output = _absolute_path(request.output_ref)
        settings = {"output": {"include_metadata": True}, "privacy_preprocess": {}}
        written: dict[str, int] = {item.scenario_ref: 0 for item in plan.scenarios}
        sequence = 0
        diagnostic: str | None = None
        cancelled = False

        def observe(kind: ObservationKind, payload) -> None:
            nonlocal sequence
            self._observe(run, sequence, kind, payload)
            sequence += 1

        def boundary() -> None:
            if self._cancel_requested(run):
                raise _Cancelled()

        def on_row(scenario_ref: str, example: dict, passed: bool, writer) -> None:
            if writer.write(_WrittenRow(example)) is not True:
                raise OSError("write")
            written[scenario_ref] += 1
            observe(
                ObservationKind.DATA_STAGE_GATE_EVALUATED,
                DataStageGateEvaluatedPayloadV1(scenario_ref, request.mode.value, passed),
            )
            observe(ObservationKind.DATA_ROW_WRITTEN, DataRowWrittenPayloadV1(scenario_ref, written[scenario_ref]))
            boundary()

        try:
            with StreamingResultWriter(output, settings) as writer:
                try:
                    if request.mode is DataMode.GENERATE:
                        self._generate(plan, client, writer, on_row, boundary, observe)
                    else:
                        self._improve(plan, client, writer, on_row, boundary, observe)
                except _Cancelled:
                    cancelled = True
                except OSError:
                    diagnostic = DIAGNOSTIC_OUTPUT_FAILED
                except Exception as error:  # noqa: BLE001 - classified to a closed code, text discarded
                    diagnostic = _classify_failure(error)
        except OSError:
            diagnostic = DIAGNOSTIC_OUTPUT_FAILED
        rows_written = sum(written.values())
        artifacts: tuple[VerifiedArtifact, ...] = ()
        if rows_written > 0:
            try:
                artifacts = (
                    _verified_artifact(DATASET_JSONL_ROLE, output),
                    _verified_artifact(DATASET_METADATA_ROLE, _sidecar_path(output)),
                )
            except OSError:
                artifacts = ()
                diagnostic = DIAGNOSTIC_OUTPUT_FAILED
        if cancelled or self._cancel_requested(run):
            state, diagnostic = DataRunState.CANCELLED, DIAGNOSTIC_CANCELLED
        elif diagnostic is not None:
            state = DataRunState.PARTIALLY_SUCCEEDED if rows_written > 0 else DataRunState.FAILED
        elif rows_written == plan.rows_requested:
            state = DataRunState.SUCCEEDED
        else:
            state, diagnostic = DataRunState.PARTIALLY_SUCCEEDED, DIAGNOSTIC_GENERATION_FAILED
        scenarios = tuple(
            DataScenarioOutcome(item.scenario_ref, item.rows, written[item.scenario_ref]) for item in plan.scenarios
        )

        def finalize(result: DataResult) -> DataResult:
            return DataResult(
                result.schema_version, result.run, state, result.mode, result.backend, result.model,
                result.rows_requested, rows_written, scenarios, artifacts, diagnostic,
            )

        self._transition(run, finalize)
        with self._lock:
            self._cancel_flags.discard(run.run_id)

    def _generate(self, plan: DataPlan, client, writer, on_row, boundary, observe) -> None:
        from SynthChat.engine import ImprovementEngine
        from SynthChat.generator import SynthChatGenerator

        request = plan.request
        ports = self._ports
        stage_validation = request.max_iterations > 1
        engine = None
        if stage_validation:
            engine = ImprovementEngine(
                llm_client=client,
                rubrics_dir=ports.rubrics_dir,
                config_path=ports.config_dir / "validation.yaml",
                logger=_LOGGER,
                enable_interactions=False,
            )
        generator = SynthChatGenerator(
            config_dir=ports.config_dir,
            scenarios_dir=ports.scenarios_dir,
            rubrics_dir=ports.rubrics_dir,
            llm_client=client,
            engine=engine,
            environment_validator=None,
            enable_stage_validation=stage_validation,
            logger=_LOGGER,
        )
        for target in plan.scenarios:
            boundary()
            scenario = generator.scenario_loader.get_scenario(target.scenario_ref)
            if not isinstance(scenario, dict):
                raise _closed(DataOperationCode.SCENARIO_INVALID)
            produced = 0
            for _ in range(target.rows):
                result = generator.generate_single(
                    scenario_key=target.scenario_ref,
                    scenario=scenario,
                    max_iterations=request.max_iterations,
                    randomize_params=True,
                )
                on_row(target.scenario_ref, result.example, bool(result.success), writer)
                produced += 1
            observe(ObservationKind.DATA_SCENARIO_COMPLETED, DataScenarioCompletedPayloadV1(target.scenario_ref, produced))

    def _improve(self, plan: DataPlan, client, writer, on_row, boundary, observe) -> None:
        from SynthChat.engine import ImprovementEngine

        request = plan.request
        ports = self._ports
        engine = ImprovementEngine(
            llm_client=client,
            rubrics_dir=ports.rubrics_dir,
            config_path=ports.config_dir / "validation.yaml",
            logger=_LOGGER,
            enable_interactions=False,
        )
        (target,) = plan.scenarios
        rows = _read_dataset_rows(_absolute_path(request.input_dataset_ref or ""))
        boundary()
        produced = 0
        for example in rows:
            result = engine.run(
                example=example, rubric_keys=list(request.rubric_refs), max_iterations=request.max_iterations,
            )
            on_row(target.scenario_ref, result.improved_example, bool(result.passed), writer)
            produced += 1
        observe(ObservationKind.DATA_SCENARIO_COMPLETED, DataScenarioCompletedPayloadV1(target.scenario_ref, produced))

    # -- show, cancel, list -------------------------------------------------------------

    def show(self, run: DataRunRef) -> DataOutcome:
        if type(run) is not DataRunRef:
            raise TypeError("run must be exact DataRunRef")
        return self._require_head(run).result.outcome()

    def cancel(self, run: DataRunRef, reason: str) -> DataOutcome:
        if type(run) is not DataRunRef:
            raise TypeError("run must be exact DataRunRef")
        with self._lock:
            head = self._require_head(run)
            state = head.result.state
            if state.terminal:
                raise _closed(DataOperationCode.CANCEL_INELIGIBLE)
            if state is DataRunState.CANCEL_REQUESTED:
                return head.result.outcome()
            self._cancel_flags.add(run.run_id)
            updated = self._transition(
                run,
                lambda result: _with_state(result, DataRunState.CANCEL_REQUESTED),
                note=redact(reason, max_string_bytes=256),
            )
        return updated.result.outcome()

    def list(self, request: DataListRequest) -> DataPage:
        if type(request) is not DataListRequest:
            raise TypeError("request must be exact DataListRequest")
        prefix = f"run/{_project_digest(request.project_ref)}/"
        after = _decode_cursor(request.cursor)
        if after is not None and not after.startswith(prefix):
            raise _closed(DataOperationCode.CURSOR_INVALID)
        try:
            page = self._records.list_page(partition=_DATA, prefix=prefix, after_key=after, limit=request.limit)
        except Exception:
            raise _integrity() from None
        outcomes = []
        for stored in page.records:
            head = self._decode_head(stored)
            if head.result.run.project_ref != request.project_ref:
                raise _integrity()
            outcomes.append(head.result.outcome())
        next_cursor = None if page.next_cursor is None else _encode_cursor(page.next_cursor)
        return DataPage(request, tuple(outcomes), next_cursor, page.truncated)

    # -- read-only dataset queries ------------------------------------------------------

    def datasets(self, request: DatasetListRequest) -> DatasetPage:
        if type(request) is not DatasetListRequest:
            raise TypeError("request must be exact DatasetListRequest")
        directory = _absolute_path(request.directory_ref)
        if not directory.is_dir():
            raise _closed(DataOperationCode.DATASET_MISSING)
        after = _decode_cursor(request.cursor)
        try:
            names = sorted(
                path.name for path in directory.iterdir()
                if path.is_file() and path.suffix == ".jsonl" and (after is None or path.name > after)
            )
        except OSError:
            raise _closed(DataOperationCode.DATASET_MISSING) from None
        page = names[: request.limit]
        truncated = len(names) > request.limit
        descriptors = []
        for name in page:
            path = directory / name
            sidecar = _sidecar_path(path)
            try:
                size = path.stat().st_size
            except OSError:
                raise _closed(DataOperationCode.DATASET_MISSING) from None
            descriptors.append(DatasetDescriptor(str(path), size, str(sidecar) if sidecar.is_file() else None))
        return DatasetPage(
            request, tuple(descriptors), _encode_cursor(page[-1]) if truncated else None, truncated,
        )

    def validate(self, request: DatasetValidateRequest) -> ValidationReport:
        if type(request) is not DatasetValidateRequest:
            raise TypeError("request must be exact DatasetValidateRequest")
        path = _absolute_path(request.dataset_ref)
        if not path.is_file():
            raise _closed(DataOperationCode.DATASET_MISSING)
        offset = _decode_cursor(request.cursor)
        skip = 0
        if offset is not None:
            if not offset.isdigit():
                raise _closed(DataOperationCode.CURSOR_INVALID)
            skip = int(offset)
        findings: list[ValidationFinding] = []
        checked = 0
        valid = 0
        last_line = skip
        more = False
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if line_number <= skip or not line.strip():
                        continue
                    if checked >= request.limit:
                        more = True
                        break
                    checked += 1
                    last_line = line_number
                    code = _validate_row(line)
                    if code is None:
                        valid += 1
                    else:
                        findings.append(ValidationFinding(line_number, code))
        except OSError:
            raise _closed(DataOperationCode.DATASET_MISSING) from None
        return ValidationReport(
            request, checked, valid, tuple(findings), _encode_cursor(str(last_line)) if more else None, more,
        )


def _with_state(result: DataResult, state: DataRunState) -> DataResult:
    return DataResult(
        result.schema_version, result.run, state, result.mode, result.backend, result.model,
        result.rows_requested, result.rows_written, result.scenarios, result.artifacts,
        result.diagnostic_code, result.usage,
    )


def _classify_failure(error: BaseException) -> str:
    """Map a generation failure to a closed diagnostic code; the message is never kept."""
    try:
        from shared.llm.exceptions import LLMError
    except Exception:  # pragma: no cover - shared.llm is a hard dependency of SynthChat
        return DIAGNOSTIC_GENERATION_FAILED
    if isinstance(error, (LLMError, ConnectionError, TimeoutError)):
        return DIAGNOSTIC_BACKEND_UNAVAILABLE
    return DIAGNOSTIC_GENERATION_FAILED


def _validate_row(line: str) -> ValidationFindingCode | None:
    try:
        row = json.loads(line)
    except ValueError:
        return ValidationFindingCode.MALFORMED_JSON
    if type(row) is not dict:
        return ValidationFindingCode.ROW_NOT_OBJECT
    if "_meta" in row:
        return ValidationFindingCode.LEGACY_METADATA_ROW
    if "conversations" not in row:
        return ValidationFindingCode.CONVERSATIONS_MISSING
    conversations = row["conversations"]
    if type(conversations) is not list or not conversations:
        return ValidationFindingCode.CONVERSATIONS_INVALID
    for message in conversations:
        if type(message) is not dict or type(message.get("role")) is not str or not message["role"]:
            return ValidationFindingCode.CONVERSATIONS_INVALID
        if "content" not in message and "tool_calls" not in message:
            return ValidationFindingCode.CONVERSATIONS_INVALID
    return None


def _plus_seconds(timestamp: str, seconds: int) -> str:
    from datetime import datetime, timedelta, timezone

    value = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
    moment = datetime.fromisoformat(value) + timedelta(seconds=seconds)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_data_operations(
    *,
    records: DurableRecordStorePort,
    streams: DurableStreamStorePort,
    clock: ClockPort,
    ports: ReferenceDataPortsV1,
) -> ReferenceDataOperationsV1:
    """The ``DataOperations`` implementation over the public host ports."""
    return ReferenceDataOperationsV1(records=records, streams=streams, clock=clock, ports=ports)


__all__ = [
    "DATA_DURABLE_EVENT_BUDGET",
    "LLMClientFactoryPort",
    "LOCAL_BACKENDS",
    "METERED_BACKENDS",
    "ReferenceDataOperationsV1",
    "ReferenceDataPortsV1",
    "build_data_operations",
    "data_run_ref",
]
