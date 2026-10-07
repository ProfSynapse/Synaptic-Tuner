"""One-shot training and chat helpers for the minimal Modal consumer example."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from synaptic_tuner.api.v1.planning import ResolvedTrainingRequest, TrainingPlan
from synaptic_tuner.api.v1.providers import ProviderRef
from synaptic_tuner.api.v1.results import TrainingRunRef, VerifiedArtifact
from synaptic_tuner.api.v1.runs_facade import RunsAPI
from synaptic_tuner.api.v1.training_facade import (
    TrainingAPI,
    TrainingPreflight,
    TrainingRequest,
    TrainingStart,
)
from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    parse_canonical_object,
    safe_ref,
)
from tuner.inference.run_chat import (
    PreparedModelIdentity,
    RunChatRuntime,
    open_run_chat,
)

from .storage import ModalChatStorage

_MAX_PROMPT_BYTES = 64 * 1024
_MAX_REQUEST_BYTES = 1024 * 1024
_MAX_RESPONSE_BYTES = 64 * 1024


class ModalChatConsumerError(RuntimeError):
    """Closed failure; the durable one-shot claim remains authoritative."""


def _canonical_payload(value: object) -> bytes:
    if type(value) is not bytes:
        raise TypeError("exact canonical evidence bytes required")
    document = parse_canonical_object(value, name="consumer evidence")
    if canonical_bytes(document) != value:
        raise ValueError("consumer evidence is not canonical")
    return value


def _catalog(storage: ModalChatStorage, name: str):
    _require_storage(storage)
    return storage.catalog(name, encode=_canonical_payload, decode=_canonical_payload)


def _require_storage(storage: ModalChatStorage) -> None:
    if type(storage) is not ModalChatStorage:
        raise TypeError("exact open ModalChatStorage required")
    try:
        storage.attempts.resolve("consumer-storage-readiness")
    except Exception:
        raise TypeError("exact open ModalChatStorage required") from None


@dataclass(frozen=True, slots=True)
class SubmittedTraining:
    request: TrainingRequest
    resolved: ResolvedTrainingRequest
    plan: TrainingPlan
    preflight: TrainingPreflight
    start: TrainingStart
    evidence: bytes


@dataclass(frozen=True, slots=True)
class SavedChatTurn:
    run: TrainingRunRef
    artifacts: tuple[VerifiedArtifact, ...]
    model: PreparedModelIdentity
    response: str
    evidence: bytes
    context_evidence: bytes


def submit_training_once(
    training: TrainingAPI,
    storage: ModalChatStorage,
    *,
    attempt_ref: str,
    request_json: str,
    provider: ProviderRef,
) -> SubmittedTraining:
    """Claim permanently before provider planning, then save returned evidence."""
    if type(training) is not TrainingAPI or type(provider) is not ProviderRef:
        raise TypeError("exact training API and provider required")
    _require_storage(storage)
    attempt_ref = safe_ref(attempt_ref, "attempt_ref")
    if type(request_json) is not str or not request_json:
        raise ValueError("request_json must be a nonempty string")
    if len(request_json) > _MAX_REQUEST_BYTES:
        raise ValueError("request_json exceeds the example bound")
    try:
        request_bytes = request_json.encode("utf-8")
    except UnicodeError:
        raise ValueError("request_json must be valid UTF-8") from None
    if len(request_bytes) > _MAX_REQUEST_BYTES:
        raise ValueError("request_json exceeds the example bound")
    storage.attempts.claim(
        attempt_ref,
        canonical_bytes(
            {
                "schema_version": "synaptic-modal-chat-training-attempt/v1",
                "request_json_sha256": hashlib.sha256(request_bytes).hexdigest(),
                "provider": provider.to_dict(),
            }
        ),
    )
    try:
        request = training.load(request_json)
        resolved = training.resolve(request)
        if (
            type(request) is not TrainingRequest
            or type(resolved) is not ResolvedTrainingRequest
            or request.canonical_json != request_json
        ):
            raise ValueError
        plan = training.plan(resolved, provider)
        preflight = training.preflight(plan)
        start = training.start(plan, preflight)
        if (
            type(plan) is not TrainingPlan
            or type(preflight) is not TrainingPreflight
            or type(start) is not TrainingStart
        ):
            raise ValueError
        evidence = canonical_bytes(
            {
                "schema_version": "synaptic-modal-chat-training-result/v1",
                "attempt_ref": attempt_ref,
                "request": {
                    "request_id": request.request_id,
                    "project_ref": request.project_ref,
                },
                "resolved": resolved.to_dict(),
                "plan": plan.to_dict(),
                "preflight": preflight.to_dict(),
                "run": start.run.to_dict(),
                "accepted": start.accepted,
            }
        )
        _catalog(storage, "training-results").publish_if_absent(attempt_ref, evidence)
        if start.accepted is not True:
            raise ValueError
        return SubmittedTraining(request, resolved, plan, preflight, start, evidence)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ModalChatConsumerError("modal_chat_training_failed") from None


def chat_once(
    runs: RunsAPI,
    storage: ModalChatStorage,
    *,
    attempt_ref: str,
    run: TrainingRunRef,
    runtime: RunChatRuntime,
    prompt: str,
) -> SavedChatTurn:
    """Claim one remote open, save its reply, and always close its context."""
    if type(runs) is not RunsAPI or type(run) is not TrainingRunRef:
        raise TypeError("exact runs API and training run required")
    _require_storage(storage)
    if type(prompt) is not str or not prompt:
        raise ValueError("prompt must be a nonempty string")
    if len(prompt) > _MAX_PROMPT_BYTES:
        raise ValueError("prompt exceeds the example bound")
    try:
        prompt_bytes = prompt.encode("utf-8")
    except UnicodeError:
        raise ValueError("prompt must be valid UTF-8") from None
    if len(prompt_bytes) > _MAX_PROMPT_BYTES:
        raise ValueError("prompt exceeds the example bound")
    attempt_ref = safe_ref(attempt_ref, "attempt_ref")
    storage.attempts.claim(
        attempt_ref,
        canonical_bytes(
            {
                "schema_version": "synaptic-modal-chat-session-attempt/v1",
                "run": run.to_dict(),
                "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            }
        ),
    )
    try:
        with open_run_chat(runs, run, runtime=runtime) as prepared:
            response = prepared.session.chat(prompt)
            if type(response.message) is not str:
                raise ValueError
            if len(response.message) > _MAX_RESPONSE_BYTES:
                raise ValueError
            try:
                response_bytes = response.message.encode("utf-8")
            except UnicodeError:
                raise ValueError from None
            if len(response_bytes) > _MAX_RESPONSE_BYTES:
                raise ValueError
            evidence = canonical_bytes(
                {
                    "schema_version": "synaptic-modal-chat-session-result/v1",
                    "attempt_ref": attempt_ref,
                    "run": prepared.run.to_dict(),
                    "artifacts": [item.to_dict() for item in prepared.artifacts],
                    "model": {
                        "model_ref": prepared.model.model_ref,
                        "model_revision": prepared.model.model_revision,
                        "tokenizer_revision": prepared.model.tokenizer_revision,
                        "model_kind": prepared.model.model_kind,
                    },
                    "response": response.message,
                }
            )
            saved = (
                prepared.run,
                prepared.artifacts,
                prepared.model,
                response.message,
                evidence,
            )
            _catalog(storage, "chat-results").publish_if_absent(attempt_ref, evidence)
        context_evidence = canonical_bytes(
            {
                "schema_version": "synaptic-modal-chat-context-result/v1",
                "attempt_ref": attempt_ref,
                "run": run.to_dict(),
                "context_closed": True,
                "provider_shutdown_proof": False,
            }
        )
        _catalog(storage, "chat-contexts").publish_if_absent(
            attempt_ref, context_evidence
        )
        return SavedChatTurn(
            *saved,
            context_evidence,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ModalChatConsumerError("modal_chat_session_failed") from None


__all__ = [
    "ModalChatConsumerError",
    "SavedChatTurn",
    "SubmittedTraining",
    "chat_once",
    "submit_training_once",
]
