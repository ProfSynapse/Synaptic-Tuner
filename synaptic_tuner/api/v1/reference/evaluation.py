"""Evaluation reference composition: ``EvaluationOperations`` over ``Evaluator/``.

Location: ``synaptic_tuner/api/v1/reference/evaluation.py``.

``EvaluationOperations`` (``api/v1/evaluation_facade.py``: plan, preflight,
start, show, cancel, list, result, observations) is implemented here by
``ReferenceEvaluationOperationsV1`` over the engine's scenario loader
(``Evaluator/config_loader.py``), runner (``Evaluator/runner.py``) and
reporting projection (``Evaluator/reporting.py``). Nothing of the legacy CLI
coupling (§10 of ``docs/architecture/api-facade-v1.md``) is inherited: the
scenario root, the backend connection, the judge and the artifact sink are all
injected through ``ReferenceEvaluationPortsV1``; no ambient credential is read,
no ``Evaluator/results/`` path is derived, no ``SystemExit`` escapes.

Durable layout (partition -> key -> document):

- ``evaluation`` / ``run/<project_digest>/<run_key>``: the run's head document
  on the record port, holding only the current run record, its revision and
  digest and the sequence and rolling chain digest of the lifecycle event that
  last changed its state. The head is bounded by
  ``EVALUATION_HEAD_MAXIMUM_BYTES`` and never grows with the number of cases.
- ``evaluation`` / same key on the stream port: the run's lifecycle events
  (``created``, ``started``, ``cancel_requested``, ``finished``). At most
  ``EVALUATION_EVENT_BUDGET`` (12) durable events are ever appended per run;
  the writer asserts the budget before every append. Progress
  (``cases_scored``) is a head compare-and-swap at scenario boundaries and
  appends no event.
- ``observation`` / ``evaluation/<project_digest>/<run_key>``: the public
  ``evaluation_*`` observation records (§4.2), one stream per run, read back
  through ``observations``. They carry no authority; a host may truncate them.

Cancellation is cooperative at scenario boundaries: ``cancel`` moves the head
to ``cancel_requested`` and the runner, which re-reads the head after every
scenario, stops before the next one and finishes as ``cancelled`` with the
partial ``evaluation_results`` artifact listed. Results and traces are written
only to the injected ``EvaluationArtifactSinkPort``; the engine tree is never
touched. A local backend claims no spend effect, so a run has zero effects and
zero grants; a backend whose requirement carries ``paid_effect`` is refused
with ``backend_unmetered`` before any client is opened (§9, metering lands in
slice 11), and a backend the registry does not know is refused with
``backend_unavailable``.

Consumed by ``synaptic_tuner/api/v1/reference/composition.py`` and exported
lazily through ``synaptic_tuner/api/v1/reference/__init__.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from threading import RLock
from typing import Protocol

from synaptic_tuner.api.v1.evaluation_facade import (
    EVALUATION_PLAN_SCHEMA_VERSION,
    EVALUATION_RESULT_SCHEMA_VERSION,
    EvaluationListRequest,
    EvaluationModelRef,
    EvaluationOperationCode,
    EvaluationOperationError,
    EvaluationOutcome,
    EvaluationPage,
    EvaluationPlan,
    EvaluationPreflight,
    EvaluationRequest,
    EvaluationResult,
    EvaluationResultRequest,
    EvaluationRunRef,
    EvaluationRunState,
    EvaluationStart,
    EvaluationVerdictCounts,
    JudgeVerdict,
    ScoreV1,
)
from synaptic_tuner.api.v1.observations import (
    OBSERVATION_SCHEMA_VERSION,
    EvaluationCaseScoredPayloadV1,
    EvaluationCaseStartedPayloadV1,
    EvaluationStageCompletedPayloadV1,
    ObservationFamily,
    ObservationKind,
    ObservationPage,
    ObservationRecordV1,
    ObservationsRequest,
    ObservationStreamRef,
)
from synaptic_tuner.api.v1.ports import (
    DurableRecordStorePort,
    DurableStreamStorePort,
    SecretResolverPort,
    StoragePartition,
    StoredRecordV1,
    StoredStreamPageV1,
)
from synaptic_tuner.api.v1.results import VerifiedArtifact
from synaptic_tuner.api.v1.secrets import SecretRef
from synaptic_tuner.api.v1.training_facade import AuthorizationRequirement
from Evaluator.config_loader import ConfigLoader
from Evaluator.reporting import (
    RUN_TRACE_SCHEMA_VERSION,
    aggregate_stats,
    build_run_payload,
    record_to_dict,
    record_trace_to_dict,
)
from Evaluator.runner import EvaluationRecord, evaluate_cases
from tuner.execution.foundation_v2.canonical import digest_text, domain_digest
from tuner.execution.redaction import redact

from .provider_family import require_methods


# --- constants ---------------------------------------------------------------------

EVALUATION_HEAD_SCHEMA = "synaptic-reference-evaluation-head/v1"
EVALUATION_RECORD_SCHEMA = "synaptic-reference-evaluation-record/v1"
EVALUATION_EVENT_SCHEMA = "synaptic-reference-evaluation-event/v1"
EVALUATION_CHAIN_DOMAIN = "synaptic-reference-evaluation-chain/v1"
EVALUATION_RUN_DOMAIN = "synaptic-reference-evaluation-run/v1"
EVALUATION_PROJECT_DOMAIN = "synaptic-reference-evaluation-project-key/v1"
EVALUATION_SCENARIO_DOMAIN = "synaptic-reference-evaluation-scenarios/v1"
EVALUATION_EVENT_BUDGET = 12
EVALUATION_HEAD_MAXIMUM_BYTES = 64 * 1024
EVALUATION_ARTIFACT_MAXIMUM_BYTES = 64 * 1024 * 1024
EVALUATION_PREFLIGHT_SECONDS = 300
EVALUATION_RESULTS_ROLE = "evaluation_results"
EVALUATION_TRACE_ROLE = "evaluation_trace"
EVALUATION_START_OPERATION = "evaluation.start"
EVALUATION_JUDGE_OPERATION = "evaluation.judge"
HISTORY_PAGE_LIMIT = 200
APPEND_ATTEMPTS = 4
LOCAL_HTTP_BACKENDS = frozenset({"ollama", "lmstudio", "vllm"})
PAID_BACKENDS = frozenset({"openrouter", "openai_responses"})

_EVALUATION = StoragePartition.EVALUATION.value
_OBSERVATION = StoragePartition.OBSERVATION.value
_TERMINAL_STATES = frozenset({
    EvaluationRunState.SUCCEEDED,
    EvaluationRunState.PARTIALLY_SUCCEEDED,
    EvaluationRunState.FAILED,
    EvaluationRunState.CANCELLED,
})
_EVENT_KINDS = frozenset({"created", "started", "cancel_requested", "finished"})
_HEAD_KEYS = frozenset({"schema_version", "revision", "sequence", "chain_digest", "record_digest", "record"})
_RECORD_KEYS = frozenset({
    "schema_version", "run", "plan", "state", "cases_scored", "artifacts",
    "diagnostic_code", "scores", "verdicts",
})
_ENTRY_KEYS = frozenset({
    "schema_version", "revision", "previous_sequence", "previous_chain_digest",
    "event", "record_digest", "chain_digest",
})
_EVENT_KEYS = frozenset({"kind", "occurred_at", "state"})
_RUN_ID_RE = re.compile(r"^ev-[0-9a-f]{32}$")
_SINK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,127}$")
_REDACT_DEPTH = 32
_REDACT_STRING_BYTES = 1024 * 1024
_REDACT_ITEMS = 65536


def _error(code: EvaluationOperationCode) -> EvaluationOperationError:
    return EvaluationOperationError(code)


def _integrity() -> EvaluationOperationError:
    return _error(EvaluationOperationCode.INTEGRITY_ERROR)


# --- host ports ---------------------------------------------------------------------


class EvaluationBackendPort(Protocol):
    """One evaluation backend a host offers under a name.

    ``requirement`` describes the authorization a start would need; a
    requirement with ``paid_effect`` marks the backend as metered and is
    refused until spend metering lands (slice 11). ``open`` returns an
    ``Evaluator`` ``BackendClient`` bound to the requested model.
    """

    def requirement(self) -> AuthorizationRequirement: ...

    def open(self, model: EvaluationModelRef): ...


class EvaluationBackendRegistryPort(Protocol):
    def resolve(self, backend: str) -> EvaluationBackendPort | None: ...


class EvaluationJudgePort(Protocol):
    """An LLM-as-judge a host offers; ``open`` returns an ``Evaluator`` ``JudgeValidator``."""

    def requirement(self) -> AuthorizationRequirement: ...

    def open(self): ...


class EvaluationArtifactSinkPort(Protocol):
    """Where result and trace bytes go: a bounded sink per ``(run_id, role)``.

    ``open`` returns a sink with ``write(chunk)``, ``finish() -> sha256`` and
    ``abort()``, the same shape as ``ArtifactSpoolPortV1`` sinks. The engine
    never derives an output path of its own.
    """

    def open(self, run_id: str, role: str, maximum_bytes: int): ...


@dataclass(frozen=True, slots=True)
class ReferenceEvaluationPortsV1:
    """What a host injects to compose the evaluation family."""

    config_root: Path
    backends: EvaluationBackendRegistryPort
    artifacts: EvaluationArtifactSinkPort
    judge: EvaluationJudgePort | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.config_root, Path):
            raise TypeError("config_root must be a Path")
        require_methods(self.backends, "resolve")
        require_methods(self.artifacts, "open")
        if self.judge is not None:
            require_methods(self.judge, "requirement", "open")


# --- reference port implementations ------------------------------------------------


class EvaluationBackendRegistryV1:
    """``EvaluationBackendRegistryPort`` over a fixed name -> backend mapping."""

    __slots__ = ("_backends",)

    def __init__(self, backends: dict[str, EvaluationBackendPort]) -> None:
        if type(backends) is not dict:
            raise TypeError("backends must be an exact dict")
        for name, backend in backends.items():
            if type(name) is not str or not name:
                raise TypeError("backend names must be non-empty strings")
            require_methods(backend, "requirement", "open")
        self._backends = dict(backends)

    def resolve(self, backend: str) -> EvaluationBackendPort | None:
        return self._backends.get(backend)


class LocalHttpEvaluatorBackendV1:
    """A local HTTP inference server (ollama, lmstudio, vllm) with explicit connection values.

    Every connection value is passed by the host; nothing is read from the
    process environment. An optional API key is a ``SecretRef`` resolved
    through the host's ``SecretResolverPort`` at ``open`` time only.
    """

    __slots__ = ("_backend", "_host", "_port", "_scheme", "_api_key", "_secrets", "_timeout", "_retries")

    def __init__(
        self,
        backend: str,
        *,
        host: str,
        port: int,
        scheme: str = "http",
        api_key: SecretRef | None = None,
        secrets: SecretResolverPort | None = None,
        timeout: float = 60.0,
        retries: int = 2,
    ) -> None:
        if backend not in LOCAL_HTTP_BACKENDS:
            raise ValueError("backend must be one of the local HTTP backends")
        if type(host) is not str or not host or type(port) is not int or not 0 < port < 65536:
            raise ValueError("host and port must name a local server")
        if scheme not in ("http", "https"):
            raise ValueError("scheme must be http or https")
        if api_key is not None:
            if type(api_key) is not SecretRef:
                raise TypeError("api_key must be a SecretRef")
            if secrets is None:
                raise ValueError("a SecretResolverPort is required to resolve api_key")
            require_methods(secrets, "resolve")
        self._backend = backend
        self._host = host
        self._port = port
        self._scheme = scheme
        self._api_key = api_key
        self._secrets = secrets
        self._timeout = float(timeout)
        self._retries = int(retries)

    def requirement(self) -> AuthorizationRequirement:
        return AuthorizationRequirement(EVALUATION_START_OPERATION, False)

    def open(self, model: EvaluationModelRef):
        # Imported here so composing a host never loads the HTTP client stack.
        from Evaluator.client_factory import create_client
        from Evaluator.config import LMStudioSettings, OllamaSettings, VLLMSettings

        settings_type = {
            "ollama": OllamaSettings, "lmstudio": LMStudioSettings, "vllm": VLLMSettings,
        }[self._backend]
        api_key = None
        if self._api_key is not None:
            api_key = self._secrets.resolve(self._api_key)  # type: ignore[union-attr]
        settings = settings_type(
            model=model.model_ref, scheme=self._scheme, api_key=api_key,
            host=self._host, port=self._port,
        )
        return create_client(self._backend, settings, timeout=self._timeout, retries=self._retries)


class UnmeteredPaidBackendV1:
    """A paid backend (openrouter, openai_responses) declared but not yet metered.

    Its requirement carries ``paid_effect`` so every start is refused with
    ``backend_unmetered`` before a client is opened; ``open`` refuses too.
    """

    __slots__ = ("_backend",)

    def __init__(self, backend: str) -> None:
        if backend not in PAID_BACKENDS:
            raise ValueError("backend must be one of the paid backends")
        self._backend = backend

    def requirement(self) -> AuthorizationRequirement:
        return AuthorizationRequirement(EVALUATION_START_OPERATION, True)

    def open(self, model: EvaluationModelRef):
        raise _error(EvaluationOperationCode.BACKEND_UNMETERED)


class _FileSink:
    __slots__ = ("_path", "_handle", "_hash", "_maximum", "_size", "_closed")

    def __init__(self, path: Path, maximum_bytes: int) -> None:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._path = path
        self._handle = os.fdopen(descriptor, "wb")
        self._hash = hashlib.sha256()
        self._maximum = maximum_bytes
        self._size = 0
        self._closed = False

    def write(self, chunk: bytes) -> None:
        if self._closed or type(chunk) is not bytes:
            raise ValueError("artifact sink is not writable")
        if self._size + len(chunk) > self._maximum:
            raise ValueError("artifact sink bound exceeded")
        self._handle.write(chunk)
        self._hash.update(chunk)
        self._size += len(chunk)

    def finish(self) -> str:
        if self._closed:
            raise ValueError("artifact sink already closed")
        self._closed = True
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        return self._hash.hexdigest()

    def abort(self) -> None:
        if not self._closed:
            self._closed = True
            self._handle.close()
        try:
            self._path.unlink()
        except FileNotFoundError:
            pass


class DirectoryEvaluationArtifactSinkV1:
    """``EvaluationArtifactSinkPort`` writing ``<root>/<run_id>/<role>.json``.

    ``root`` is the host's choice. Identifiers are restricted to a flat safe
    alphabet so no run or role can escape the root, files are created with
    ``O_EXCL`` so an existing artifact is never overwritten, and every sink is
    bounded by the caller's ``maximum_bytes``.
    """

    __slots__ = ("_root",)

    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path):
            raise TypeError("root must be a Path")
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def open(self, run_id: str, role: str, maximum_bytes: int):
        if type(run_id) is not str or _SINK_ID_RE.fullmatch(run_id) is None:
            raise ValueError("run_id must be a bounded safe identifier")
        if type(role) is not str or _SINK_ID_RE.fullmatch(role) is None:
            raise ValueError("role must be a bounded safe identifier")
        if type(maximum_bytes) is not int or maximum_bytes < 1:
            raise ValueError("maximum_bytes must be a positive integer")
        directory = self._root / run_id
        directory.mkdir(parents=True, exist_ok=True)
        return _FileSink(directory / f"{role}.json", maximum_bytes)


# --- canonical documents ------------------------------------------------------------


def _dump(document: dict[str, object], maximum: int = EVALUATION_HEAD_MAXIMUM_BYTES) -> bytes:
    try:
        encoded = json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise _integrity() from None
    if len(encoded) > maximum:
        raise _integrity()
    return encoded


def _load(raw: bytes, schema_version: str) -> dict[str, object]:
    if type(raw) is not bytes or not raw or len(raw) > EVALUATION_HEAD_MAXIMUM_BYTES:
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
    return domain_digest(EVALUATION_PROJECT_DOMAIN, project_ref.encode("utf-8"))


def _run_key(run_id: str) -> str:
    return domain_digest(EVALUATION_RUN_DOMAIN, run_id.encode("utf-8"))


def _run_id(plan: EvaluationPlan) -> str:
    return "ev-" + domain_digest(EVALUATION_RUN_DOMAIN, plan.plan_fingerprint.encode("ascii"))[:32]


def _head_key(run: EvaluationRunRef) -> str:
    return f"run/{_project_digest(run.project_ref)}/{_run_key(run.run_id)}"


def _observation_key(run: EvaluationRunRef) -> str:
    return f"evaluation/{_project_digest(run.project_ref)}/{_run_key(run.run_id)}"


def _stream_ref(run: EvaluationRunRef) -> ObservationStreamRef:
    return ObservationStreamRef(ObservationFamily.EVALUATION, run.project_ref, run.run_id)


@dataclass(frozen=True, slots=True)
class _RunRecord:
    """The durable run record; the head document carries exactly one of these."""

    run: EvaluationRunRef
    plan: EvaluationPlan
    state: EvaluationRunState
    cases_scored: int
    artifacts: tuple[VerifiedArtifact, ...]
    diagnostic_code: str | None
    scores: tuple[ScoreV1, ...]
    verdicts: EvaluationVerdictCounts

    def encode(self) -> dict[str, object]:
        return {
            "schema_version": EVALUATION_RECORD_SCHEMA,
            "run": self.run.to_dict(),
            "plan": self.plan.to_dict(),
            "state": self.state.value,
            "cases_scored": self.cases_scored,
            "artifacts": [item.to_dict() for item in self.artifacts],
            "diagnostic_code": self.diagnostic_code,
            "scores": [item.to_dict() for item in self.scores],
            "verdicts": self.verdicts.to_dict(),
        }

    def digest(self) -> str:
        return domain_digest(EVALUATION_RECORD_SCHEMA, _dump(self.encode()))

    def outcome(self) -> EvaluationOutcome:
        return EvaluationOutcome(
            self.run, self.state, self.plan.cases_total, self.cases_scored,
            self.artifacts, self.diagnostic_code,
        )

    def replace(self, **changes: object) -> "_RunRecord":
        values = {
            "run": self.run, "plan": self.plan, "state": self.state,
            "cases_scored": self.cases_scored, "artifacts": self.artifacts,
            "diagnostic_code": self.diagnostic_code, "scores": self.scores,
            "verdicts": self.verdicts,
        }
        values.update(changes)
        return _RunRecord(**values)  # type: ignore[arg-type]

    @classmethod
    def decode(cls, value: object) -> "_RunRecord":
        if type(value) is not dict or set(value) != _RECORD_KEYS:
            raise _integrity()
        if value["schema_version"] != EVALUATION_RECORD_SCHEMA:
            raise _integrity()
        try:
            run = EvaluationRunRef.from_dict(value["run"])
            plan = EvaluationPlan.from_dict(value["plan"])
            state = EvaluationRunState(value["state"])
            artifacts = value["artifacts"]
            scores = value["scores"]
            if type(artifacts) is not list or type(scores) is not list:
                raise ValueError
            record = cls(
                run, plan, state, value["cases_scored"],  # type: ignore[arg-type]
                tuple(VerifiedArtifact.from_dict(item) for item in artifacts),
                value["diagnostic_code"],  # type: ignore[arg-type]
                tuple(ScoreV1.from_dict(item) for item in scores),
                EvaluationVerdictCounts.from_dict(value["verdicts"]),  # type: ignore[arg-type]
            )
            record.outcome()
        except EvaluationOperationError:
            raise
        except Exception:
            raise _integrity() from None
        if _RUN_ID_RE.fullmatch(record.run.run_id) is None or record.run.run_id != _run_id(record.plan):
            raise _integrity()
        if record.run.project_ref != record.plan.request.project_ref:
            raise _integrity()
        return record


@dataclass(frozen=True, slots=True)
class _Head:
    """A validated head: the record plus the store revision and the chain position."""

    record: _RunRecord
    revision: int
    sequence: int
    chain_digest: str
    raw: bytes


def _event_digest(event: dict[str, object]) -> str:
    return domain_digest(EVALUATION_EVENT_SCHEMA, _dump(event))


def _chain_digest(previous: str | None, event_digest: str, record_digest: str) -> str:
    return domain_digest(
        EVALUATION_CHAIN_DOMAIN,
        _dump({"previous": previous, "event": event_digest, "record": record_digest}),
    )


def _verdict(record: EvaluationRecord) -> tuple[JudgeVerdict, float]:
    """Map one runner record to the closed verdict vocabulary and a finite score.

    A transport or validation error, or a judge that broke, is
    ``inconclusive`` (§7.1 Q4); otherwise the runner's pass/fail decides.
    """
    judge = record.judge
    judge_broke = judge is not None and getattr(judge.judge_result, "error", None) is not None
    score = record.score
    if score is None and judge is not None and not judge_broke:
        values = [item.score for item in judge.judge_result.scores if isinstance(item.score, (int, float))]
        if values:
            score = sum(values) / len(values)
    if record.error is not None:
        return JudgeVerdict.INCONCLUSIVE, 0.0
    if judge_broke:
        return JudgeVerdict.INCONCLUSIVE, float(score) if score is not None else 0.0
    passed = record.passed
    if score is None:
        score = 1.0 if passed else 0.0
    return (JudgeVerdict.PASSED if passed else JudgeVerdict.FAILED), float(score)


def _redacted(value: object) -> object:
    """Bounded, secret-redacted copy of one record projection (per record, not per run)."""
    return json.loads(
        redact(value, max_depth=_REDACT_DEPTH, max_string_bytes=_REDACT_STRING_BYTES, max_items=_REDACT_ITEMS)
    )


def _shift(timestamp: str, seconds: int) -> str:
    moment = datetime.fromisoformat(timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp)
    moment = moment.astimezone(timezone.utc) + timedelta(seconds=seconds)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


# --- scenario resolution ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ResolvedScenarios:
    """The plan's cases grouped by scenario reference, in request order."""

    groups: tuple[tuple[str, tuple[object, ...]], ...]
    scenario_digest: str

    @property
    def cases_total(self) -> int:
        return sum(len(cases) for _, cases in self.groups)


class _ScenarioResolver:
    """Loads scenarios from the injected ``config_root`` and nothing else."""

    __slots__ = ("_root",)

    def __init__(self, config_root: Path) -> None:
        self._root = config_root

    def _scenario_path(self, ref: str) -> Path:
        parts = ref.split("/")
        if (
            ref != ref.strip()
            or ref.startswith("/")
            or "\\" in ref
            or "://" in ref
            or any(part in ("", ".", "..") for part in parts)
        ):
            raise _error(EvaluationOperationCode.SCENARIO_INVALID)
        scenarios = (self._root / "scenarios").resolve()
        path = (scenarios / ref).resolve()
        if not path.is_relative_to(scenarios) or not path.is_file():
            raise _error(EvaluationOperationCode.SCENARIO_INVALID)
        return path

    def _preset_filters(self, preset: str | None) -> tuple[list[str], list[str], bytes | None]:
        if preset is None:
            return [], [], None
        run_file = self._root / "eval_run.yaml"
        if not run_file.is_file():
            raise _error(EvaluationOperationCode.SCENARIO_INVALID)
        raw = run_file.read_bytes()
        try:
            import yaml

            document = yaml.safe_load(raw.decode("utf-8")) or {}
        except Exception:
            raise _error(EvaluationOperationCode.SCENARIO_INVALID) from None
        presets = document.get("presets") if type(document) is dict else None
        if type(presets) is not dict or preset not in presets:
            raise _error(EvaluationOperationCode.SCENARIO_INVALID)
        try:
            run_config = ConfigLoader(self._root).load_eval_run(preset)
        except Exception:
            raise _error(EvaluationOperationCode.SCENARIO_INVALID) from None
        return list(run_config.tag_filter), list(run_config.exclude_tags), raw

    def resolve(self, request: EvaluationRequest) -> _ResolvedScenarios:
        paths = {ref: self._scenario_path(ref) for ref in request.scenario_refs}
        preset_tags, exclude_tags, run_bytes = self._preset_filters(request.preset)
        tag_filter = list(request.tags) + [tag for tag in preset_tags if tag not in request.tags]
        loader = ConfigLoader(self._root)
        groups: list[tuple[str, tuple[object, ...]]] = []
        remaining = request.case_limit
        for ref in request.scenario_refs:
            try:
                cases = loader.load_all_scenarios([ref], tag_filter or None, exclude_tags or None)
            except Exception:
                raise _error(EvaluationOperationCode.SCENARIO_INVALID) from None
            for index, case in enumerate(cases):
                if not case.case_id:
                    case.case_id = f"case-{index + 1}"
            if remaining is not None:
                cases = cases[:remaining]
                remaining -= len(cases)
            if cases:
                groups.append((ref, tuple(cases)))
        digest = domain_digest(
            EVALUATION_SCENARIO_DOMAIN,
            _dump(
                {
                    "preset": request.preset,
                    "eval_run": None if run_bytes is None else hashlib.sha256(run_bytes).hexdigest(),
                    "scenarios": {ref: hashlib.sha256(path.read_bytes()).hexdigest() for ref, path in paths.items()},
                },
                maximum=EVALUATION_ARTIFACT_MAXIMUM_BYTES,
            ),
        )
        resolved = _ResolvedScenarios(tuple(groups), digest)
        if resolved.cases_total < 1:
            raise _error(EvaluationOperationCode.SCENARIO_INVALID)
        return resolved


# --- observation writer -------------------------------------------------------------


class _ObservationWriter:
    """Appends ``evaluation_*`` observation records for one run to the stream port."""

    __slots__ = ("_streams", "_key", "_stream", "_clock", "_next")

    def __init__(self, streams: DurableStreamStorePort, run: EvaluationRunRef, clock) -> None:
        self._streams = streams
        self._key = _observation_key(run)
        self._stream = _stream_ref(run)
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


# --- operations ---------------------------------------------------------------------


class ReferenceEvaluationOperationsV1:
    """``EvaluationOperations`` over the durable stores, the scenario root and the injected ports."""

    __slots__ = ("_records", "_streams", "_clock", "_ports", "_resolver", "_validated", "_lock")

    def __init__(
        self,
        *,
        records: DurableRecordStorePort,
        streams: DurableStreamStorePort,
        clock,
        ports: ReferenceEvaluationPortsV1,
    ) -> None:
        require_methods(records, "create", "read", "compare_and_swap", "put_if_absent", "list_page")
        require_methods(streams, "append", "read_page")
        require_methods(clock, "now")
        if type(ports) is not ReferenceEvaluationPortsV1:
            raise TypeError("exact ReferenceEvaluationPortsV1 required")
        self._records = records
        self._streams = streams
        self._clock = clock
        self._ports = ports
        self._resolver = _ScenarioResolver(ports.config_root)
        self._validated: dict[str, _Head] = {}
        self._lock = RLock()

    # -- head documents ------------------------------------------------------------

    def _read_stored(self, key: str) -> StoredRecordV1 | None:
        try:
            stored = self._records.read(partition=_EVALUATION, key=key)
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
            page = _read_page(self._streams, _EVALUATION, key, cursor, HISTORY_PAGE_LIMIT)
            for entry in page.entries:
                if entry.sequence > until_sequence:
                    return found
                found.append(self._decode_entry(entry.sequence, entry.canonical))
            if not page.truncated or page.next_cursor is None or page.next_cursor >= until_sequence:
                return found
            cursor = page.next_cursor

    @staticmethod
    def _decode_entry(sequence: int, raw: bytes) -> dict[str, object]:
        document = _load(raw, EVALUATION_EVENT_SCHEMA)
        if set(document) != _ENTRY_KEYS:
            raise _integrity()
        event = document["event"]
        if type(event) is not dict or set(event) != _EVENT_KEYS or event["kind"] not in _EVENT_KINDS:
            raise _integrity()
        if type(document["revision"]) is not int or document["revision"] < 1:
            raise _integrity()
        try:
            EvaluationRunState(event["state"])
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
        document = _load(stored.canonical, EVALUATION_HEAD_SCHEMA)
        if set(document) != _HEAD_KEYS:
            raise _integrity()
        revision, sequence, chain = document["revision"], document["sequence"], document["chain_digest"]
        if type(revision) is not int or type(sequence) is not int or revision < 1 or sequence < 1:
            raise _integrity()
        if revision != stored.revision or type(chain) is not str:
            raise _integrity()
        record = _RunRecord.decode(document["record"])
        if record.digest() != document["record_digest"] or _head_key(record.run) != key:
            raise _integrity()
        entries = self._entries(key, sequence)
        if len(entries) != sequence or len(entries) > EVALUATION_EVENT_BUDGET:
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
        if last["revision"] > revision:
            raise _integrity()
        head = _Head(record, revision, sequence, chain, stored.canonical)
        with self._lock:
            self._validated[key] = head
        return head

    def _head(self, run: EvaluationRunRef) -> _Head | None:
        key = _head_key(run)
        stored = self._read_stored(key)
        return None if stored is None else self._validate(key, stored)

    def _require_head(self, run: EvaluationRunRef) -> _Head:
        head = self._head(run)
        if head is None:
            raise _error(EvaluationOperationCode.RUN_MISSING)
        return head

    @staticmethod
    def _head_document(record: _RunRecord, *, revision: int, sequence: int, chain: str) -> bytes:
        return _dump(
            {
                "schema_version": EVALUATION_HEAD_SCHEMA,
                "revision": revision,
                "sequence": sequence,
                "chain_digest": chain,
                "record_digest": record.digest(),
                "record": record.encode(),
            }
        )

    def _append_event(
        self, key: str, *, revision: int, kind: str, record: _RunRecord, previous: _Head | None
    ) -> tuple[int, str]:
        """Append one lifecycle event; asserts the per-run durable event budget."""
        previous_sequence = None if previous is None else previous.sequence
        previous_chain = None if previous is None else previous.chain_digest
        sequence = (previous_sequence or 0) + 1
        if sequence > EVALUATION_EVENT_BUDGET:
            raise _integrity()
        event = {"kind": kind, "occurred_at": self._clock.now(), "state": record.state.value}
        record_digest = record.digest()
        chain = _chain_digest(previous_chain, _event_digest(event), record_digest)
        canonical = _dump(
            {
                "schema_version": EVALUATION_EVENT_SCHEMA,
                "revision": revision,
                "previous_sequence": previous_sequence,
                "previous_chain_digest": previous_chain,
                "event": event,
                "record_digest": record_digest,
                "chain_digest": chain,
            }
        )
        try:
            appended = self._streams.append(
                partition=_EVALUATION, stream_key=key, sequence=sequence, canonical=canonical
            )
        except Exception:
            raise _integrity() from None
        if appended is True:
            return sequence, chain
        if appended is False:
            raise _error(EvaluationOperationCode.STATE_CONFLICT)
        raise _integrity()

    def _create(self, record: _RunRecord) -> _Head | None:
        """Genesis: event 1 then the head at revision 1; ``None`` when another writer won."""
        key = _head_key(record.run)
        sequence, chain = self._append_event(key, revision=1, kind="created", record=record, previous=None)
        raw = self._head_document(record, revision=1, sequence=sequence, chain=chain)
        try:
            created = self._records.create(partition=_EVALUATION, key=key, canonical=raw)
        except Exception:
            raise _integrity() from None
        if created is not True:
            return None
        head = _Head(record, 1, sequence, chain, raw)
        with self._lock:
            self._validated[key] = head
        return head

    def _transition(self, head: _Head, record: _RunRecord, kind: str) -> _Head:
        """Append the event, then commit through the head compare-and-swap."""
        key = _head_key(record.run)
        revision = head.revision + 1
        sequence, chain = self._append_event(key, revision=revision, kind=kind, record=record, previous=head)
        raw = self._head_document(record, revision=revision, sequence=sequence, chain=chain)
        try:
            swapped = self._records.compare_and_swap(
                partition=_EVALUATION, key=key, expected_revision=head.revision, canonical=raw
            )
        except Exception:
            raise _integrity() from None
        if swapped is not True:
            raise _error(EvaluationOperationCode.STATE_CONFLICT)
        updated = _Head(record, revision, sequence, chain, raw)
        with self._lock:
            self._validated[key] = updated
        return updated

    def _progress(self, head: _Head, record: _RunRecord) -> _Head:
        """Head-only progress update: no event, same chain position."""
        key = _head_key(record.run)
        revision = head.revision + 1
        raw = self._head_document(record, revision=revision, sequence=head.sequence, chain=head.chain_digest)
        try:
            swapped = self._records.compare_and_swap(
                partition=_EVALUATION, key=key, expected_revision=head.revision, canonical=raw
            )
        except Exception:
            raise _integrity() from None
        if swapped is not True:
            raise _error(EvaluationOperationCode.STATE_CONFLICT)
        updated = _Head(record, revision, head.sequence, head.chain_digest, raw)
        with self._lock:
            self._validated[key] = updated
        return updated

    # -- backends ------------------------------------------------------------------

    def _backend(self, name: str) -> EvaluationBackendPort:
        backend = self._ports.backends.resolve(name)
        if backend is None:
            raise _error(EvaluationOperationCode.BACKEND_UNAVAILABLE)
        require_methods(backend, "requirement", "open")
        return backend

    def _requirements(self, backend: EvaluationBackendPort) -> tuple[AuthorizationRequirement, ...]:
        items = [backend.requirement()]
        if self._ports.judge is not None:
            items.append(self._ports.judge.requirement())
        if any(type(item) is not AuthorizationRequirement for item in items):
            raise TypeError("backend requirement must be exact AuthorizationRequirement")
        return tuple(items)

    # -- verbs -----------------------------------------------------------------------

    def plan(self, request: EvaluationRequest) -> EvaluationPlan:
        self._backend(request.backend)
        resolved = self._resolver.resolve(request)
        return EvaluationPlan(EVALUATION_PLAN_SCHEMA_VERSION, request, resolved.cases_total, resolved.scenario_digest)

    def preflight(self, plan: EvaluationPlan) -> EvaluationPreflight:
        checked_at = self._clock.now()
        expires_at = _shift(checked_at, EVALUATION_PREFLIGHT_SECONDS)
        codes: list[str] = []
        authorization: tuple[AuthorizationRequirement, ...] = ()
        try:
            backend = self._backend(plan.request.backend)
        except EvaluationOperationError as error:
            codes.append(error.code.value)
        else:
            requirements = self._requirements(backend)
            paid = tuple(item for item in requirements if item.paid_effect)
            if paid:
                codes.append(EvaluationOperationCode.BACKEND_UNMETERED.value)
                authorization = paid
            try:
                current = self._resolver.resolve(plan.request)
            except EvaluationOperationError as error:
                codes.append(error.code.value)
            else:
                if current.scenario_digest != plan.scenario_digest or current.cases_total != plan.cases_total:
                    codes.append(EvaluationOperationCode.SCENARIO_INVALID.value)
        return EvaluationPreflight(
            plan.plan_fingerprint, not codes, checked_at, expires_at, authorization, tuple(dict.fromkeys(codes)),
        )

    def start(self, plan: EvaluationPlan) -> EvaluationStart:
        run = EvaluationRunRef(_run_id(plan), plan.request.project_ref)
        if self._head(run) is not None:
            return EvaluationStart(run, True)
        backend = self._backend(plan.request.backend)
        if any(item.paid_effect for item in self._requirements(backend)):
            raise _error(EvaluationOperationCode.BACKEND_UNMETERED)
        resolved = self._resolver.resolve(plan.request)
        if resolved.scenario_digest != plan.scenario_digest or resolved.cases_total != plan.cases_total:
            raise _error(EvaluationOperationCode.SCENARIO_INVALID)
        genesis = _RunRecord(
            run, plan, EvaluationRunState.PLANNED, 0, (), None, (), EvaluationVerdictCounts(0, 0, 0),
        )
        head = self._create(genesis)
        if head is None:
            return EvaluationStart(run, True)
        head = self._transition(head, genesis.replace(state=EvaluationRunState.RUNNING), "started")
        self._execute(head, backend, resolved)
        return EvaluationStart(run, True)

    def _execute(self, head: _Head, backend: EvaluationBackendPort, resolved: _ResolvedScenarios) -> None:
        run = head.record.run
        request = head.record.plan.request
        observations = _ObservationWriter(self._streams, run, self._clock)
        records: list[EvaluationRecord] = []
        verdicts: list[tuple[JudgeVerdict, float]] = []
        cancelled = False
        failure: str | None = None
        try:
            try:
                client = backend.open(request.model)
            except Exception:
                raise _error(EvaluationOperationCode.BACKEND_UNAVAILABLE) from None
            judge = None
            if self._ports.judge is not None:
                try:
                    judge = self._ports.judge.open()
                except Exception:
                    raise _error(EvaluationOperationCode.JUDGE_FAILED) from None
            for ref, cases in resolved.groups:
                head = self._require_head(run)
                if head.record.state is EvaluationRunState.CANCEL_REQUESTED:
                    cancelled = True
                    break
                if head.record.state is not EvaluationRunState.RUNNING:
                    raise _error(EvaluationOperationCode.STATE_CONFLICT)
                for case in cases:
                    case_ref = f"{ref}#{case.case_id}"  # type: ignore[attr-defined]
                    observations.emit(
                        ObservationKind.EVALUATION_CASE_STARTED, EvaluationCaseStartedPayloadV1(case_ref)
                    )
                    (record,) = evaluate_cases([case], client, judge_validator=judge)  # type: ignore[list-item]
                    verdict, score = _verdict(record)
                    records.append(record)
                    verdicts.append((verdict, score))
                    observations.emit(
                        ObservationKind.EVALUATION_CASE_SCORED,
                        EvaluationCaseScoredPayloadV1(case_ref, verdict.value, score),
                    )
                current = self._require_head(run)
                head = self._progress(current, current.record.replace(cases_scored=len(records)))
                observations.emit(
                    ObservationKind.EVALUATION_STAGE_COMPLETED,
                    EvaluationStageCompletedPayloadV1(ref, len(records)),
                )
        except EvaluationOperationError as error:
            failure = error.code.value
        except Exception:
            failure = EvaluationOperationCode.INTEGRITY_ERROR.value

        artifacts: tuple[VerifiedArtifact, ...] = ()
        if failure is None:
            try:
                artifacts = self._write_artifacts(run, request, records, cancelled)
            except EvaluationOperationError as error:
                failure = error.code.value
            except Exception:
                failure = EvaluationOperationCode.INTEGRITY_ERROR.value

        counts = EvaluationVerdictCounts(
            sum(1 for verdict, _ in verdicts if verdict is JudgeVerdict.PASSED),
            sum(1 for verdict, _ in verdicts if verdict is JudgeVerdict.FAILED),
            sum(1 for verdict, _ in verdicts if verdict is JudgeVerdict.INCONCLUSIVE),
        )
        scores: list[ScoreV1] = []
        if verdicts:
            scores.append(ScoreV1("composite", sum(score for _, score in verdicts) / len(verdicts)))
            gated = aggregate_stats(records).get("quality_gated_normalized_score")
            if isinstance(gated, (int, float)):
                scores.append(ScoreV1("quality_gated", float(gated)))

        if failure is not None:
            state, diagnostic = EvaluationRunState.FAILED, failure
        elif cancelled:
            state, diagnostic = EvaluationRunState.CANCELLED, None
        elif records and all(record.error is not None for record in records):
            state, diagnostic = EvaluationRunState.FAILED, EvaluationOperationCode.BACKEND_UNAVAILABLE.value
        elif counts.inconclusive:
            state, diagnostic = EvaluationRunState.PARTIALLY_SUCCEEDED, None
        else:
            state, diagnostic = EvaluationRunState.SUCCEEDED, None

        for _ in range(APPEND_ATTEMPTS):
            head = self._require_head(run)
            if head.record.state in _TERMINAL_STATES:
                return
            final = head.record.replace(
                state=state, cases_scored=len(records), artifacts=artifacts,
                diagnostic_code=diagnostic, scores=tuple(scores), verdicts=counts,
            )
            try:
                self._transition(head, final, "finished")
                return
            except EvaluationOperationError as error:
                if error.code is not EvaluationOperationCode.STATE_CONFLICT:
                    raise
        raise _error(EvaluationOperationCode.STATE_CONFLICT)

    def _write_artifacts(
        self,
        run: EvaluationRunRef,
        request: EvaluationRequest,
        records: list[EvaluationRecord],
        cancelled: bool,
    ) -> tuple[VerifiedArtifact, ...]:
        metadata = {
            "run_id": run.run_id,
            "project_ref": run.project_ref,
            "backend": request.backend,
            "model": request.model.model_ref,
            "model_revision": request.model.model_revision,
            "scenario_refs": list(request.scenario_refs),
            "preset": request.preset,
            "tags": list(request.tags),
            "case_limit": request.case_limit,
            "partial": cancelled,
        }
        payload = build_run_payload(records, metadata, generated_at=self._clock.now())
        payload["records"] = [_redacted(record_to_dict(record)) for record in records]
        trace = {
            "schema_version": RUN_TRACE_SCHEMA_VERSION,
            "run_id": run.run_id,
            "records": [_redacted(record_trace_to_dict(record)) for record in records],
        }
        return (
            self._write_artifact(run, EVALUATION_RESULTS_ROLE, payload),
            self._write_artifact(run, EVALUATION_TRACE_ROLE, trace),
        )

    def _write_artifact(self, run: EvaluationRunRef, role: str, document: dict[str, object]) -> VerifiedArtifact:
        try:
            encoded = json.dumps(
                document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            ).encode("utf-8")
        except (TypeError, ValueError):
            raise _integrity() from None
        if len(encoded) > EVALUATION_ARTIFACT_MAXIMUM_BYTES:
            raise _integrity()
        sink = self._ports.artifacts.open(run.run_id, role, EVALUATION_ARTIFACT_MAXIMUM_BYTES)
        require_methods(sink, "write", "finish", "abort")
        try:
            sink.write(encoded)
            digest = sink.finish()
        except Exception:
            sink.abort()
            raise _integrity() from None
        if digest != hashlib.sha256(encoded).hexdigest():
            sink.abort()
            raise _integrity()
        return VerifiedArtifact(role, digest, len(encoded))

    def show(self, run: EvaluationRunRef) -> EvaluationOutcome:
        return self._require_head(run).record.outcome()

    def cancel(self, run: EvaluationRunRef, reason: str) -> EvaluationOutcome:
        for _ in range(APPEND_ATTEMPTS):
            head = self._require_head(run)
            record = head.record
            if record.state in _TERMINAL_STATES:
                raise _error(EvaluationOperationCode.CANCEL_INELIGIBLE)
            if record.state is EvaluationRunState.CANCEL_REQUESTED:
                return record.outcome()
            requested = record.replace(state=EvaluationRunState.CANCEL_REQUESTED)
            try:
                return self._transition(head, requested, "cancel_requested").record.outcome()
            except EvaluationOperationError as error:
                if error.code is not EvaluationOperationCode.STATE_CONFLICT:
                    raise
        raise _error(EvaluationOperationCode.STATE_CONFLICT)

    def list(self, request: EvaluationListRequest) -> EvaluationPage:
        prefix = f"run/{_project_digest(request.project_ref)}/"
        after_key = None
        if request.cursor is not None:
            try:
                digest_text(request.cursor, "cursor")
            except ValueError:
                raise _error(EvaluationOperationCode.CURSOR_INVALID) from None
            after_key = prefix + request.cursor
            if self._read_stored(after_key) is None:
                raise _error(EvaluationOperationCode.CURSOR_INVALID)
        try:
            page = self._records.list_page(
                partition=_EVALUATION, prefix=prefix, after_key=after_key, limit=request.limit
            )
        except Exception:
            raise _integrity() from None
        outcomes = []
        for stored in page.records:
            if type(stored) is not StoredRecordV1 or not stored.key.startswith(prefix):
                raise _integrity()
            outcomes.append(self._validate(stored.key, stored).record.outcome())
        next_cursor = None
        if page.truncated:
            if type(page.next_cursor) is not str or not page.next_cursor.startswith(prefix):
                raise _integrity()
            next_cursor = page.next_cursor[len(prefix):]
        return EvaluationPage(request, tuple(outcomes), next_cursor, page.truncated is True)

    def result(self, request: EvaluationResultRequest) -> EvaluationResult:
        record = self._require_head(request.run).record
        if record.state not in _TERMINAL_STATES:
            raise _error(EvaluationOperationCode.RESULT_UNAVAILABLE)
        return EvaluationResult(
            EVALUATION_RESULT_SCHEMA_VERSION, record.run, record.state, record.plan.request.backend,
            record.plan.request.model, record.plan.cases_total, record.cases_scored, record.scores,
            record.verdicts, record.artifacts, record.diagnostic_code, None,
        )

    def observations(self, request: ObservationsRequest) -> ObservationPage:
        run = EvaluationRunRef(request.stream.entity_id, request.stream.project_ref)
        self._require_head(run)
        page = _read_page(self._streams, _OBSERVATION, _observation_key(run), request.after_sequence, request.limit)
        records = []
        for entry in page.entries:
            try:
                document = json.loads(entry.canonical.decode("utf-8"))
                record = ObservationRecordV1.from_dict(document)
            except Exception:
                raise _integrity() from None
            if record.sequence != entry.sequence or record.stream != request.stream:
                raise _integrity()
            records.append(record)
        return ObservationPage(request, tuple(records), page.next_cursor, page.truncated is True)


def compose_reference_evaluation(
    *,
    records: DurableRecordStorePort,
    streams: DurableStreamStorePort,
    clock,
    evaluation: ReferenceEvaluationPortsV1,
) -> ReferenceEvaluationOperationsV1:
    """The ``EvaluationOperations`` implementation over the host stores and evaluation ports."""
    return ReferenceEvaluationOperationsV1(records=records, streams=streams, clock=clock, ports=evaluation)


__all__ = [
    "EVALUATION_ARTIFACT_MAXIMUM_BYTES",
    "EVALUATION_EVENT_BUDGET",
    "EVALUATION_HEAD_MAXIMUM_BYTES",
    "EVALUATION_RESULTS_ROLE",
    "EVALUATION_TRACE_ROLE",
    "DirectoryEvaluationArtifactSinkV1",
    "EvaluationArtifactSinkPort",
    "EvaluationBackendPort",
    "EvaluationBackendRegistryPort",
    "EvaluationBackendRegistryV1",
    "EvaluationJudgePort",
    "LocalHttpEvaluatorBackendV1",
    "ReferenceEvaluationOperationsV1",
    "ReferenceEvaluationPortsV1",
    "UnmeteredPaidBackendV1",
    "compose_reference_evaluation",
]
