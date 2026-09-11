from contextlib import contextmanager
import os
import threading

import pytest

from Evaluator.protocols import BackendResponse
from Evaluator.chat_session import ChatSession, ChatSessionPolicy
from examples.modal_chat.consumer import (
    ModalChatConsumerError,
    chat_once,
    submit_training_once,
)
from examples.modal_chat.storage import AttemptAlreadyClaimed, ModalChatStorage
from synaptic_tuner.api.v1.planning import (
    ProviderPlanRef,
    ResolvedTrainingRequest,
    TrainingPlan,
    TrainingPlanBasisV1,
)
from synaptic_tuner.api.v1.providers import ProviderRef
from synaptic_tuner.api.v1.results import TrainingRunRef, VerifiedArtifact
from synaptic_tuner.api.v1.runs_facade import RunsAPI
from synaptic_tuner.api.v1.training_facade import (
    AuthorizationRequirement,
    TrainingAPI,
    TrainingPreflight,
    TrainingRequest,
    TrainingStart,
)
from tuner.inference.retrieved_model import ROLES
from tuner.inference.run_chat import PreparedModelIdentity, PreparedRunChat


class _Clock:
    def now(self):
        return "2026-09-11T12:00:00Z"


class _Training:
    def __init__(self, *, ready=True):
        self.calls = []
        self.request = TrainingRequest("request", "project", '{"method":"sft"}')
        self.resolved = ResolvedTrainingRequest(
            "synaptic-resolved-training-request/v1",
            "request",
            "project",
            *(character * 64 for character in "abcde"),
        )
        self.plan_value = TrainingPlan(
            "synaptic-training-plan/v2",
            TrainingPlanBasisV1.from_resolved(self.resolved),
            ProviderPlanRef("f" * 64),
        )
        self.preflight_value = TrainingPreflight(
            self.plan_value.plan_fingerprint,
            ready,
            "2026-09-11T11:59:00Z",
            "2026-09-11T12:01:00Z",
            (AuthorizationRequirement("start", True, 1000, "USD"),) if ready else (),
            () if ready else ("not_ready",),
        )
        self.start_value = TrainingStart(TrainingRunRef("run", "project"), True)

    def load(self, value):
        self.calls.append("load")
        return self.request

    def resolve(self, value):
        self.calls.append("resolve")
        return self.resolved

    def plan(self, resolved, provider):
        self.calls.append("plan")
        return self.plan_value

    def preflight(self, plan):
        self.calls.append("preflight")
        return self.preflight_value

    def start(self, plan, preflight):
        self.calls.append("start")
        return self.start_value


def _storage(tmp_path):
    os.chmod(tmp_path, 0o700)
    return ModalChatStorage(tmp_path / "consumer.sqlite3", "example")


def test_training_claim_precedes_provider_plan_and_saves_canonical_result(tmp_path):
    operations = _Training()
    training = TrainingAPI(operations, clock=_Clock())
    with _storage(tmp_path) as storage:
        original_plan = operations.plan

        def plan_after_claim(resolved, provider):
            assert storage.attempts.resolve("train-1") is not None
            return original_plan(resolved, provider)

        operations.plan = plan_after_claim
        result = submit_training_once(
            training,
            storage,
            attempt_ref="train-1",
            request_json=operations.request.canonical_json,
            provider=ProviderRef("modal", "chat-a10"),
        )
        assert result.start.run == TrainingRunRef("run", "project")
        retained = storage.catalog(
            "training-results", encode=lambda value: value, decode=lambda value: value
        ).resolve("train-1")
        assert retained == result.evidence
        with pytest.raises(AttemptAlreadyClaimed):
            submit_training_once(
                training,
                storage,
                attempt_ref="train-1",
                request_json=operations.request.canonical_json,
                provider=ProviderRef("modal", "chat-a10"),
            )
    assert operations.calls == ["load", "resolve", "plan", "preflight", "start"]


def test_not_ready_preflight_never_reaches_paid_start(tmp_path):
    operations = _Training(ready=False)
    with _storage(tmp_path) as storage, pytest.raises(ModalChatConsumerError):
        submit_training_once(
            TrainingAPI(operations, clock=_Clock()),
            storage,
            attempt_ref="train-denied",
            request_json=operations.request.canonical_json,
            provider=ProviderRef("modal", "chat-a10"),
        )
    assert operations.calls == ["load", "resolve", "plan", "preflight"]


@pytest.mark.parametrize("cleanup_failure", (False, True))
@pytest.mark.parametrize("response_text", ("reply", "x" * 65537, "é" * 40000))
def test_chat_claims_before_open_saves_reply_and_closes(
    tmp_path, cleanup_failure, response_text
):
    response_rejected = len(response_text.encode("utf-8")) > 65536
    run = TrainingRunRef("run", "project")
    artifacts = tuple(
        VerifiedArtifact(role, str(index + 1) * 64, index + 1)
        for index, role in enumerate(ROLES)
    )
    model = PreparedModelIdentity("model", "a" * 40, "b" * 40, "full")

    class Lease:
        def __init__(self):
            self.closed = threading.Event()

        @property
        def cleanup_pending(self):
            return not self.closed.is_set()

        def close(self, **kwargs):
            self.closed.set()
            return True

    class Backend:
        def chat(self, messages):
            assert messages == ({"role": "user", "content": "hello"},)
            return BackendResponse(response_text, {}, 0.1)

    lease = Lease()
    session = ChatSession(
        Backend(),
        lease,
        ChatSessionPolicy(1, 1, 10, 1, 1024 * 1024),
    )

    class Runtime:
        opens = 0

        @contextmanager
        def open(self, runs, requested):
            self.opens += 1
            assert storage.attempts.resolve("chat-1") is not None
            with session:
                yield PreparedRunChat(session, run, artifacts, model)
            assert lease.closed.wait(0.5)
            if cleanup_failure:
                raise RuntimeError("synthetic private cleanup failure")

    runtime = Runtime()
    with _storage(tmp_path) as storage:
        if cleanup_failure or response_rejected:
            with pytest.raises(ModalChatConsumerError):
                chat_once(
                    RunsAPI(object()),
                    storage,
                    attempt_ref="chat-1",
                    run=run,
                    runtime=runtime,
                    prompt="hello",
                )
        else:
            result = chat_once(
                RunsAPI(object()),
                storage,
                attempt_ref="chat-1",
                run=run,
                runtime=runtime,
                prompt="hello",
            )
            assert result.response == "reply"
        results = storage.catalog(
            "chat-results", encode=lambda value: value, decode=lambda value: value
        ).resolve("chat-1")
        contexts = storage.catalog(
            "chat-contexts", encode=lambda value: value, decode=lambda value: value
        ).resolve("chat-1")
        assert (results is None) is response_rejected
        assert (contexts is None) is (cleanup_failure or response_rejected)
        with pytest.raises(AttemptAlreadyClaimed):
            chat_once(
                RunsAPI(object()),
                storage,
                attempt_ref="chat-1",
                run=run,
                runtime=runtime,
                prompt="hello",
            )
    assert lease.closed.wait(0.5)
    assert runtime.opens == 1


def test_oversize_prompt_is_rejected_before_claim(monkeypatch, tmp_path):
    with _storage(tmp_path) as storage:
        with pytest.raises(ValueError, match="exceeds"):
            chat_once(
                RunsAPI(object()),
                storage,
                attempt_ref="chat-large",
                run=TrainingRunRef("run", "project"),
                runtime=object(),
                prompt="x" * (64 * 1024 + 1),
            )
        assert storage.attempts.resolve("chat-large") is None


def test_closed_storage_is_rejected_before_training_or_chat_effects(tmp_path):
    operations = _Training()
    training = TrainingAPI(operations, clock=_Clock())
    storage = _storage(tmp_path)
    storage.close()
    with pytest.raises(TypeError, match="open ModalChatStorage"):
        submit_training_once(
            training,
            storage,
            attempt_ref="closed-train",
            request_json=operations.request.canonical_json,
            provider=ProviderRef("modal", "chat-a10"),
        )
    assert operations.calls == []

    class Runtime:
        opens = 0

        def open(self, runs, run):
            self.opens += 1
            raise AssertionError

    runtime = Runtime()
    with pytest.raises(TypeError, match="open ModalChatStorage"):
        chat_once(
            RunsAPI(object()),
            storage,
            attempt_ref="closed-chat",
            run=TrainingRunRef("run", "project"),
            runtime=runtime,
            prompt="hello",
        )
    assert runtime.opens == 0
