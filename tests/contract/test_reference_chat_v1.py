"""Reference ``ChatAPI`` over faked runtimes (api-facade slice 8).

Composes ``compose_reference_chat`` (and once ``compose_reference_host`` with
``ReferenceChatPortsV1``) over the public host ports: in-memory record and
stream stores, a fixed clock and a scripted runs facade. Every runtime here is
a fake at the ``RunChatRuntime`` / ``ChatRuntimePort`` boundary wrapping the
real engine ``ChatSession``; a socket guard proves no network call happens, a
``subprocess.Popen`` gate proves no vLLM process is ever started, and a
worktree scan proves nothing is written to the checkout.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading

import pytest

from Evaluator.protocols import BackendResponse
from synaptic_tuner.api.v1.chat_facade import (
    ChatAPI, ChatListRequest, ChatModelKind, ChatModelSource, ChatOpenRequest, ChatOperationCode,
    ChatOperationError, ChatSession, ChatSessionPolicyV1, ChatSessionRef, ChatSessionState, ChatTurn,
    ChatTurnRef, ChatTurnRequest, content_digest,
)
from synaptic_tuner.api.v1.execution import ExecutionGrant
from synaptic_tuner.api.v1.observations import (
    ObservationFamily, ObservationKind, ObservationsRequest, ObservationStreamRef,
)
from synaptic_tuner.api.v1.planning import (
    ProviderPlanContextV1, ResolvedTrainingRequest, TrainingPlanBasisV1,
)
from synaptic_tuner.api.v1.ports import StoragePartition
from synaptic_tuner.api.v1.providers import ProviderCapabilities, ProviderDescriptor, ProviderRef
from synaptic_tuner.api.v1.reference import (
    ChatRuntimeRegistryV1, LocalVLLMChatRuntimeV1, ProviderFamilyV1, ReferenceChatPortsV1,
    ReferenceHostPortsV1, ReferenceRequestPortsV1, compose_reference_authority, compose_reference_host,
)
from synaptic_tuner.api.v1.reference.chat import (
    CHAT_HEAD_MAXIMUM_BYTES, CHAT_OBSERVATION_BUDGET, CHAT_OBSERVATIONS_PER_TURN, LocalHostInfoV1,
    compose_reference_chat,
)
from synaptic_tuner.api.v1.reference.stores import (
    InMemoryDurableRecordStoreV1, InMemoryDurableStreamStoreV1,
)
from synaptic_tuner.api.v1.results import TrainingRunRef, VerifiedArtifact
from synaptic_tuner.api.v1.runs_facade import RunLogEntry, RunLogLevel, RunsAPI
from synaptic_tuner.api.v1.secrets import SecretRef
from synaptic_tuner.api.v1.training_facade import (
    AuthorizationRequirement, TrainingPreflight, TrainingRequest,
)
from synaptic_tuner.api.v1.usage import UsageAvailability
from tuner.execution.coordinator_v1.model import ProviderExecutionBindingV1, ProviderRunPhaseV1
from tuner.execution.fake_provider_v1 import (
    FakeArtifactV1, FakeProviderConfigV1, FakeProviderFamilyV1,
)
from tuner.execution.foundation_v2.commands import CanonicalProviderPayloadV1
from tuner.execution.foundation_v2.executors import AdapterDescriptorV1, ExecutorDescriptorV1
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.foundation_v2.references import ExecutionScopeV1
from tuner.inference.chat_session import ChatSession as EngineChatSession
from tuner.inference.chat_session import ChatSessionPolicy
from tuner.inference.owned_process import OwnedProcessError
from tuner.inference.retrieved_model import ROLES
from tuner.inference.run_chat import PreparedModelIdentity, PreparedRunChat


ROOT = Path(__file__).resolve().parents[2]
D = tuple(character * 64 for character in "123456789abcdef")
SECRET = SecretRef("env", "SYNAPTIC_REFERENCE_AUTHORITY")
PROVIDER = ProviderRef("fake", "profile")
PROJECT = "project"
RUN = TrainingRunRef("run-1", PROJECT)
NOW = "2026-09-17T00:00:00Z"
HUB = ChatModelSource("acme/qwen-sft", "a" * 40)
SECRET_VALUE = "sk-live-VERYSECRETVALUE000000000000"
_SKIP_DIRS = frozenset({".git", "__pycache__", ".pytest_cache", "node_modules", "_worktrees", ".mypy_cache"})


# --- guards ---------------------------------------------------------------------------


def _refuse_network(*args, **kwargs):
    raise AssertionError("network access is forbidden in the chat reference test")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _refuse_network)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse_network)
    monkeypatch.setattr(socket, "create_connection", _refuse_network)
    monkeypatch.setattr(socket, "getaddrinfo", _refuse_network)


@pytest.fixture(autouse=True)
def _no_vllm_process(monkeypatch):
    """No verb of the reference chat family may start a vLLM (or any) process."""
    real = subprocess.Popen

    class Popen(real):
        def __init__(self, args, *rest, **kwargs):
            text = " ".join(str(item) for item in (args if isinstance(args, (list, tuple)) else [args]))
            if "vllm" in text.lower():
                raise AssertionError(f"a vLLM process was about to start: {text[:120]}")
            super().__init__(args, *rest, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", Popen)


def _worktree_files() -> frozenset[str]:
    found = set()
    for directory, names, files in os.walk(ROOT):
        names[:] = [name for name in names if name not in _SKIP_DIRS]
        for name in files:
            found.add(os.path.join(directory, name))
    return frozenset(found)


# --- fakes -------------------------------------------------------------------------------


class Clock:
    def now(self) -> str:
        return NOW

    def now_epoch(self) -> int:
        return 150


class RecordingStreams(InMemoryDurableStreamStoreV1):
    """Records every partition appended to so the test can prove where events went."""

    def __init__(self):
        super().__init__()
        self.appended: list[tuple[str, str, int]] = []

    def append(self, *, partition, stream_key, sequence, canonical):
        self.appended.append((partition, stream_key, sequence))
        return super().append(partition=partition, stream_key=stream_key, sequence=sequence, canonical=canonical)


class Client:
    """A local ``BackendClient`` answering from a script; never touches a socket."""

    def __init__(self, reply="hello there", raw=None, block=None):
        self.reply = reply
        self.raw = {} if raw is None else raw
        self.block = block
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        if self.block is not None:
            self.block.wait(5)
        reply = self.reply(messages) if callable(self.reply) else self.reply
        return BackendResponse(message=reply, raw=self.raw, latency_s=0.0)


class Lease:
    def __init__(self, *, resolves=True):
        self.cleanup_pending = True
        self.resolves = resolves
        self.closes = 0

    def close(self, **kwargs):
        self.closes += 1
        self.cleanup_pending = not self.resolves
        return self.resolves


def _engine_policy(policy: ChatSessionPolicyV1) -> ChatSessionPolicy:
    return ChatSessionPolicy(
        policy.request_timeout_seconds, policy.idle_timeout_seconds, policy.absolute_lifetime_seconds,
        policy.max_turns, policy.max_history_bytes,
    )


class FakeRuntime:
    """``ChatRuntimePort`` over the real engine ``ChatSession`` with a scripted client and lease."""

    def __init__(self, client=None, lease=None, *, exit_error=None, prepared_model=None):
        self.client = client if client is not None else Client()
        self.lease = lease if lease is not None else Lease()
        self.exit_error = exit_error
        self.prepared_model = prepared_model or PreparedModelIdentity("example/model", "c" * 40, "c" * 40, "lora")
        self.opened = []
        self.sessions = []

    @contextmanager
    def _serve(self, policy):
        session = EngineChatSession(self.client, self.lease, _engine_policy(policy))
        self.sessions.append(session)
        with session:
            yield session
        if self.exit_error is not None:
            raise self.exit_error

    def open_model(self, source, policy):
        self.opened.append(("model", source))
        return self._serve(policy)

    def open_run(self, runs, run, policy):
        self.opened.append(("run", run, runs))
        return self._prepared(run, policy)

    @contextmanager
    def _prepared(self, run, policy):
        with self._serve(policy) as session:
            yield PreparedRunChat(
                session, run, tuple(VerifiedArtifact(role, "a" * 64, 1) for role in ROLES), self.prepared_model,
            )


class FakeRunChatRuntime:
    """A host-injected ``RunChatRuntime`` (the boundary the lead asked to fake)."""

    def __init__(self, client=None, lease=None):
        self.client = client if client is not None else Client()
        self.lease = lease if lease is not None else Lease()
        self.opened = []

    @contextmanager
    def open(self, runs, run):
        self.opened.append((runs, run))
        session = EngineChatSession(self.client, self.lease, ChatSessionPolicy(30, 600, 3600, 50, 1 << 20))
        with session:
            yield PreparedRunChat(
                session, run, tuple(VerifiedArtifact(role, "a" * 64, 1) for role in ROLES),
                PreparedModelIdentity("acme/run-model", "b" * 40, "b" * 64, "full"),
            )


class HostInfo:
    def __init__(self, supported):
        self.supported = supported

    def supports_owned_processes(self):
        return self.supported


def _policy(**changes):
    values = {
        "request_timeout_seconds": 30.0, "idle_timeout_seconds": 600.0, "absolute_lifetime_seconds": 3600.0,
        "max_turns": 50, "max_history_bytes": 1 << 20,
    }
    values.update(changes)
    return ChatSessionPolicyV1(**values)


def _open(runtime="fake", **changes):
    values = {"project_ref": PROJECT, "runtime": runtime, "policy": _policy(), "model": HUB}
    values.update(changes)
    return ChatOpenRequest(**values)


def _api(runtimes, records=None, streams=None):
    records = records if records is not None else InMemoryDurableRecordStoreV1()
    streams = streams if streams is not None else RecordingStreams()
    operations = compose_reference_chat(
        records=records, streams=streams, clock=Clock(), runs=RunsAPI(object()),
        chat=ReferenceChatPortsV1(ChatRuntimeRegistryV1(runtimes)),
    )
    return ChatAPI(operations), records, streams


def _stores(records):
    found = {}
    for partition in StoragePartition:
        page = records.list_page(partition=partition.value, prefix="", after_key=None, limit=100)
        if page.records:
            found[partition.value] = tuple(record.key for record in page.records)
    return found


def _observations(api, session, **kwargs):
    stream = ObservationStreamRef(ObservationFamily.CHAT, session.project_ref, session.session_id)
    return api.observations(ObservationsRequest(stream, **kwargs))


# --- open / turn / show / close ladder -------------------------------------------------------


def test_model_first_session_ladder_with_consecutive_request_ids_and_no_lifecycle_record():
    runtime = FakeRuntime(Client(lambda messages: f"reply to {messages[-1]['content']}"))
    api, records, streams = _api({"fake": runtime})
    before = _worktree_files()

    session = api.open(_open())
    assert session.state is ChatSessionState.READY
    assert session.model == HUB.identity() and session.run is None
    assert session.turns == 0 and session.history_bytes == 0
    assert session.ref.session_id.startswith("cs-") and session.ref.project_ref == PROJECT
    assert runtime.opened == [("model", HUB)]

    turns = [api.turn(ChatTurnRequest(session.ref, f"question {index}")) for index in range(1, 4)]
    assert [turn.ref.request_id for turn in turns] == [1, 2, 3]
    assert all(type(turn) is ChatTurn for turn in turns)
    previous = None
    for turn in turns:
        assert turn.ref.follows(previous)
        previous = turn.ref
    assert turns[0].content == "reply to question 1"
    assert turns[0].content_digest == content_digest("reply to question 1")
    assert turns[0].state is ChatSessionState.READY
    assert "usage" not in turns[0].to_dict()

    shown = api.show(session.ref)
    assert shown.state is ChatSessionState.READY and shown.turns == 3
    assert shown.history_bytes == runtime.sessions[0].state.history_bytes > 0

    # Exactly one head record per session, in the chat_session partition only.
    assert set(_stores(records)) == {StoragePartition.CHAT_SESSION.value}
    assert len(_stores(records)[StoragePartition.CHAT_SESSION.value]) == 1
    # Observations are the only stream traffic: two per turn, none per lifecycle step.
    assert {partition for partition, _, _ in streams.appended} == {StoragePartition.OBSERVATION.value}
    assert len(streams.appended) == CHAT_OBSERVATIONS_PER_TURN * 3
    page = _observations(api, session.ref)
    kinds = [record.kind for record in page.records]
    assert kinds == [ObservationKind.CHAT_TURN_STARTED, ObservationKind.CHAT_TURN_COMPLETED] * 3
    assert [record.sequence for record in page.records] == list(range(1, 7))
    assert [record.payload.request_id for record in page.records] == [1, 1, 2, 2, 3, 3]
    completed = page.records[1].payload
    assert completed.content_digest == turns[0].content_digest
    assert completed.content_bytes == len(turns[0].content.encode("utf-8"))

    closed = api.close(session.ref)
    assert closed.state is ChatSessionState.CLOSED and closed.diagnostic_code is None
    assert closed.turns == 3
    assert runtime.lease.closes == 1 and not runtime.lease.cleanup_pending
    assert api.close(session.ref) == closed  # idempotent
    with pytest.raises(ChatOperationError) as captured:
        api.turn(ChatTurnRequest(session.ref, "after close"))
    assert captured.value.code is ChatOperationCode.SESSION_CLOSED
    assert runtime.client.calls == 3
    assert _worktree_files() == before
    assert not any(".synaptic" in path for path in _worktree_files())


def test_run_first_session_binds_the_run_and_the_prepared_identity_through_run_chat_runtime(tmp_path):
    run_runtime = FakeRunChatRuntime()
    local = LocalVLLMChatRuntimeV1(
        cwd=tmp_path, environment={}, host=HostInfo(True), run_runtime=run_runtime,
    )
    api, records, streams = _api({"local": local})
    session = api.open(_open("local", model=None, run=RUN))
    assert session.run == RUN
    assert session.model.model_ref == "acme/run-model" and session.model.model_kind is ChatModelKind.FULL
    assert session.model.model_revision == "b" * 40 and session.model.tokenizer_revision == "b" * 64
    (runs, run), = run_runtime.opened
    assert type(runs) is RunsAPI and run == RUN
    turn = api.turn(ChatTurnRequest(session.ref, "hi"))
    assert turn.ref == ChatTurnRef(session.ref, 1) and turn.content == "hello there"
    assert api.close(session.ref).state is ChatSessionState.CLOSED


def test_run_first_without_an_injected_run_runtime_is_runtime_unavailable(tmp_path):
    local = LocalVLLMChatRuntimeV1(cwd=tmp_path, environment={}, host=HostInfo(True))
    api, records, streams = _api({"local": local})
    with pytest.raises(ChatOperationError) as captured:
        api.open(_open("local", model=None, run=RUN))
    assert captured.value.code is ChatOperationCode.RUNTIME_UNAVAILABLE
    (record,) = api.list(ChatListRequest(PROJECT)).sessions
    assert record.state is ChatSessionState.CLOSED and record.diagnostic_code == "runtime_unavailable"
    assert record.model is None and record.run == RUN


def test_unknown_runtime_name_is_runtime_unavailable_before_any_head_is_written():
    api, records, streams = _api({"fake": FakeRuntime()})
    with pytest.raises(ChatOperationError) as captured:
        api.open(_open("elsewhere"))
    assert captured.value.code is ChatOperationCode.RUNTIME_UNAVAILABLE
    assert _stores(records) == {} and streams.appended == []


def test_measured_usage_is_published_and_unavailable_usage_is_omitted():
    measured = FakeRuntime(Client(raw={"usage": {"prompt_tokens": 12, "completion_tokens": 7}}))
    partial = FakeRuntime(Client(raw={"usage": {"prompt_tokens": 12}}))
    api, _, _ = _api({"measured": measured, "partial": partial})
    session = api.open(_open("measured"))
    turn = api.turn(ChatTurnRequest(session.ref, "hi"))
    assert turn.usage is not None and turn.usage.availability is UsageAvailability.MEASURED
    assert (turn.usage.input_tokens, turn.usage.output_tokens) == (12, 7)
    assert "usage" in turn.to_dict()
    other = api.open(_open("partial"))
    assert api.turn(ChatTurnRequest(other.ref, "hi")).usage is None


# --- busy, bounds, missing -------------------------------------------------------------------


def test_second_in_flight_turn_is_session_busy_and_the_first_completes():
    gate = threading.Event()
    runtime = FakeRuntime(Client("slow reply", block=gate))
    api, _, streams = _api({"fake": runtime})
    session = api.open(_open())
    outcomes = {}

    def first():
        outcomes["first"] = api.turn(ChatTurnRequest(session.ref, "first"))

    worker = threading.Thread(target=first)
    worker.start()
    for _ in range(200):
        if runtime.client.calls == 1:
            break
        threading.Event().wait(0.01)
    assert runtime.client.calls == 1
    with pytest.raises(ChatOperationError) as captured:
        api.turn(ChatTurnRequest(session.ref, "second"))
    assert captured.value.code is ChatOperationCode.SESSION_BUSY
    assert api.show(session.ref).state is ChatSessionState.SERVING
    with pytest.raises(ChatOperationError) as captured:
        api.close(session.ref)
    assert captured.value.code is ChatOperationCode.SESSION_BUSY
    gate.set()
    worker.join(5)
    assert not worker.is_alive()
    assert outcomes["first"].ref.request_id == 1 and outcomes["first"].content == "slow reply"
    assert api.show(session.ref).state is ChatSessionState.READY
    second = api.turn(ChatTurnRequest(session.ref, "second"))
    assert second.ref.request_id == 2
    assert [sequence for _, _, sequence in streams.appended] == [1, 2, 3, 4]


def test_turn_beyond_the_session_bounds_is_turn_bounds_invalid_and_closes_the_session():
    runtime = FakeRuntime()
    api, _, _ = _api({"fake": runtime})
    session = api.open(_open(policy=_policy(max_turns=2)))
    assert api.turn(ChatTurnRequest(session.ref, "one")).ref.request_id == 1
    assert api.turn(ChatTurnRequest(session.ref, "two")).ref.request_id == 2
    with pytest.raises(ChatOperationError) as captured:
        api.turn(ChatTurnRequest(session.ref, "three"))
    assert captured.value.code is ChatOperationCode.TURN_BOUNDS_INVALID
    record = api.show(session.ref)
    assert record.state is ChatSessionState.CLOSED and record.diagnostic_code == "turn_bounds_invalid"
    assert record.turns == 2 and runtime.lease.closes == 1
    assert runtime.client.calls == 2

    history = api.open(_open(policy=_policy(max_history_bytes=64)))
    with pytest.raises(ChatOperationError) as captured:
        api.turn(ChatTurnRequest(history.ref, "x" * 65))
    assert captured.value.code is ChatOperationCode.TURN_BOUNDS_INVALID
    assert api.show(history.ref).state is ChatSessionState.CLOSED


def test_backend_failure_during_a_turn_is_runtime_unavailable_without_leaking_detail():
    def explode(messages):
        raise RuntimeError("private-backend-detail")

    runtime = FakeRuntime(Client(explode))
    api, _, _ = _api({"fake": runtime})
    session = api.open(_open())
    with pytest.raises(ChatOperationError) as captured:
        api.turn(ChatTurnRequest(session.ref, "hi"))
    # The engine session wraps and closes on a backend failure; the facade reports the closed session.
    assert captured.value.code in (ChatOperationCode.SESSION_CLOSED, ChatOperationCode.RUNTIME_UNAVAILABLE)
    assert captured.value.__cause__ is None and captured.value.__context__ is None
    assert "private" not in str(captured.value)
    record = api.show(session.ref)
    assert record.state is ChatSessionState.CLOSED and record.diagnostic_code == captured.value.code.value
    page = _observations(api, session.ref)
    assert [record.kind for record in page.records] == [ObservationKind.CHAT_TURN_STARTED]


def test_engine_session_bound_reached_during_a_turn_is_session_closed():
    """The engine's own history bound (user + assistant bytes) closes the session; the facade reports it."""
    runtime = FakeRuntime(Client("y" * 60))
    api, _, _ = _api({"fake": runtime})
    session = api.open(_open(policy=_policy(max_history_bytes=64)))
    with pytest.raises(ChatOperationError) as captured:
        api.turn(ChatTurnRequest(session.ref, "x" * 10))
    assert captured.value.code is ChatOperationCode.SESSION_CLOSED
    assert captured.value.__cause__ is None and captured.value.__context__ is None
    record = api.show(session.ref)
    assert record.state is ChatSessionState.CLOSED and record.diagnostic_code == "session_closed"
    assert runtime.lease.closes == 1


def test_missing_session_is_session_missing_for_every_session_verb():
    api, _, _ = _api({"fake": FakeRuntime()})
    ghost = ChatSessionRef("cs-ghost", PROJECT)
    for call in (
        lambda: api.show(ghost),
        lambda: api.close(ghost),
        lambda: api.turn(ChatTurnRequest(ghost, "hi")),
        lambda: _observations(api, ghost),
    ):
        with pytest.raises(ChatOperationError) as captured:
            call()
        assert captured.value.code is ChatOperationCode.SESSION_MISSING


# --- cleanup_unresolved and host_unsupported ----------------------------------------------------


@pytest.mark.parametrize("shape", ["lease_pending", "owned_process_error"])
def test_unresolved_cleanup_is_terminal_uncertain_and_refuses_further_turns(shape):
    if shape == "lease_pending":
        runtime = FakeRuntime(lease=Lease(resolves=False))
    else:
        runtime = FakeRuntime(exit_error=OwnedProcessError("owned process family cleanup remains unresolved"))
    api, _, _ = _api({"fake": runtime})
    session = api.open(_open())
    assert api.turn(ChatTurnRequest(session.ref, "hi")).ref.request_id == 1
    closed = api.close(session.ref)
    assert closed.state is ChatSessionState.CLEANUP_UNRESOLVED
    assert closed.diagnostic_code == "cleanup_unresolved"
    assert closed.turns == 1 and closed.model == HUB.identity()
    for _ in range(2):
        with pytest.raises(ChatOperationError) as captured:
            api.turn(ChatTurnRequest(session.ref, "again"))
        assert captured.value.code is ChatOperationCode.CLEANUP_UNRESOLVED
    assert api.close(session.ref) == closed
    assert api.show(session.ref) == closed
    assert runtime.client.calls == 1


def test_unresolved_cleanup_during_open_is_recorded_and_raised():
    class Runtime(FakeRuntime):
        @contextmanager
        def _serve(self, policy):
            from tuner.inference.model_chat import ModelChatError, ModelChatFailure, ModelChatFailureCode

            raise ModelChatError(ModelChatFailure(ModelChatFailureCode.FAILED, self.lease))
            yield  # pragma: no cover

    api, _, _ = _api({"fake": Runtime()})
    with pytest.raises(ChatOperationError) as captured:
        api.open(_open())
    assert captured.value.code is ChatOperationCode.CLEANUP_UNRESOLVED
    (record,) = api.list(ChatListRequest(PROJECT)).sessions
    assert record.state is ChatSessionState.CLEANUP_UNRESOLVED and record.diagnostic_code == "cleanup_unresolved"
    assert record.model is None and record.turns == 0


def test_non_posix_host_receives_host_unsupported_from_the_local_arm_before_any_spawn(tmp_path):
    local = LocalVLLMChatRuntimeV1(cwd=tmp_path, environment={}, host=HostInfo(False))
    api, records, streams = _api({"local": local})
    for request in (_open("local"), _open("local", model=None, run=RUN)):
        with pytest.raises(ChatOperationError) as captured:
            api.open(request)
        assert captured.value.code is ChatOperationCode.HOST_UNSUPPORTED
    sessions = api.list(ChatListRequest(PROJECT)).sessions
    assert len(sessions) == 2
    assert all(record.state is ChatSessionState.CLOSED for record in sessions)
    assert all(record.diagnostic_code == "host_unsupported" for record in sessions)
    assert streams.appended == []


def test_local_host_info_mirrors_the_engine_probe(monkeypatch):
    info = LocalHostInfoV1()
    assert info.supports_owned_processes() is (os.name == "posix" and Path("/proc/self/stat").is_file() and hasattr(os, "killpg"))
    monkeypatch.setattr(os, "name", "nt")
    assert info.supports_owned_processes() is False


def test_local_arm_model_first_goes_through_open_model_chat_without_a_real_process(tmp_path, monkeypatch):
    from tuner.inference import model_chat
    from tuner.inference.vllm_runtime import VLLMRuntimeLease

    lease = Lease()
    runtime_lease = VLLMRuntimeLease(lease, host="127.0.0.1", port=8000, served_model_name="synaptic-chat")
    starts = []

    def start(spec, **kwargs):
        starts.append((spec, kwargs))
        return runtime_lease

    class VLLMClient:
        def __init__(self, settings, **kwargs):
            self.settings = settings

        def chat(self, messages):
            return BackendResponse("served " + messages[-1]["content"], {}, 0.0)

    monkeypatch.setattr(model_chat, "start_vllm_runtime", start)
    monkeypatch.setattr(model_chat, "VLLMClient", VLLMClient)
    local = LocalVLLMChatRuntimeV1(cwd=tmp_path, environment={"HF_HOME": str(tmp_path)}, host=HostInfo(True))
    api, _, _ = _api({"local": local})
    session = api.open(_open("local"))
    assert session.model == HUB.identity()
    (spec, kwargs), = starts
    assert spec.source.model_ref == "acme/qwen-sft" and spec.source.revision == "a" * 40
    assert spec.served_model_name == "synaptic-chat"
    assert kwargs["cwd"] == tmp_path and kwargs["environment"]["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert api.turn(ChatTurnRequest(session.ref, "ping")).content == "served ping"
    assert api.close(session.ref).state is ChatSessionState.CLOSED
    assert lease.closes == 1


def test_local_arm_refuses_a_non_canonical_model_source_as_model_ineligible(tmp_path, monkeypatch):
    from tuner.inference import model_chat

    monkeypatch.setattr(model_chat, "start_vllm_runtime", lambda *a, **k: pytest.fail("started"))
    local = LocalVLLMChatRuntimeV1(cwd=tmp_path, environment={}, host=HostInfo(True))
    api, _, _ = _api({"local": local})
    missing = ChatModelSource(str(tmp_path / "absent"), "f" * 64)
    with pytest.raises(ChatOperationError) as captured:
        api.open(_open("local", model=missing))
    assert captured.value.code is ChatOperationCode.MODEL_INELIGIBLE
    (record,) = api.list(ChatListRequest(PROJECT)).sessions
    assert record.state is ChatSessionState.CLOSED and record.diagnostic_code == "model_ineligible"


# --- redaction, listing, observations paging ----------------------------------------------------------


def test_reply_content_is_redacted_and_no_exception_text_is_ever_stored():
    runtime = FakeRuntime(Client(f"token is Bearer {SECRET_VALUE} and key {SECRET_VALUE}"))
    api, records, streams = _api({"fake": runtime})
    session = api.open(_open())
    turn = api.turn(ChatTurnRequest(session.ref, "reveal"))
    assert SECRET_VALUE not in turn.content
    assert "[REDACTED]" in turn.content
    assert turn.content_digest == content_digest(turn.content)
    dump = json.dumps([_stores(records), streams.appended])
    everything = b"".join(
        record.canonical for partition in StoragePartition
        for record in records.list_page(partition=partition.value, prefix="", after_key=None, limit=100).records
    ) + b"".join(
        entry.canonical for entry in streams.read_page(
            partition=StoragePartition.OBSERVATION.value, stream_key=streams.appended[0][1], after_sequence=None, limit=100,
        ).entries
    )
    assert SECRET_VALUE.encode() not in everything and "private" not in dump

    class Boom(FakeRuntime):
        @contextmanager
        def _serve(self, policy):
            raise RuntimeError("private-backend-detail sk-live-LEAK00000000")
            yield  # pragma: no cover

    boom_api, boom_records, _ = _api({"boom": Boom()})
    with pytest.raises(ChatOperationError) as captured:
        boom_api.open(_open("boom"))
    assert captured.value.code is ChatOperationCode.RUNTIME_UNAVAILABLE
    assert captured.value.__cause__ is None and captured.value.__context__ is None
    stored = b"".join(
        record.canonical for record in boom_records.list_page(
            partition=StoragePartition.CHAT_SESSION.value, prefix="", after_key=None, limit=100,
        ).records
    )
    assert b"private-backend-detail" not in stored and b"LEAK" not in stored


def test_list_pages_sessions_by_project_and_cursor_and_the_head_stays_small():
    api, records, _ = _api({"fake": FakeRuntime()})
    opened = [api.open(_open()) for _ in range(3)]
    api.open(_open(project_ref="other"))
    first = api.list(ChatListRequest(PROJECT, limit=2))
    assert len(first.sessions) == 2 and first.truncated and first.next_cursor is not None
    second = api.list(ChatListRequest(PROJECT, cursor=first.next_cursor, limit=2))
    assert len(second.sessions) == 1 and not second.truncated and second.next_cursor is None
    listed = {record.ref for record in first.sessions + second.sessions}
    assert listed == {record.ref for record in opened}
    assert all(record.ref.project_ref == PROJECT for record in first.sessions + second.sessions)
    assert len(api.list(ChatListRequest("other")).sessions) == 1
    for cursor in ("not-a-key", "0" * 64):
        with pytest.raises(ChatOperationError) as captured:
            api.list(ChatListRequest(PROJECT, cursor=cursor))
        assert captured.value.code is ChatOperationCode.SESSION_MISSING
    for record in records.list_page(partition=StoragePartition.CHAT_SESSION.value, prefix="", after_key=None, limit=100).records:
        assert len(record.canonical) <= CHAT_HEAD_MAXIMUM_BYTES
        document = json.loads(record.canonical)
        assert set(document) == {"schema_version", "revision", "record_digest", "record"}
        assert ChatSession.from_dict(document["record"]).ref.project_ref in {PROJECT, "other"}


def test_observations_page_with_after_sequence_and_the_budget_is_two_per_turn():
    api, _, _ = _api({"fake": FakeRuntime()})
    session = api.open(_open(policy=_policy(max_turns=3)))
    for index in range(3):
        api.turn(ChatTurnRequest(session.ref, f"q{index}"))
    page = _observations(api, session.ref, limit=4)
    assert [record.sequence for record in page.records] == [1, 2, 3, 4]
    assert page.truncated and page.next_cursor == 4
    rest = _observations(api, session.ref, after_sequence=4, limit=4)
    assert [record.sequence for record in rest.records] == [5, 6] and not rest.truncated
    assert all(record.stream.family is ObservationFamily.CHAT for record in rest.records)
    assert CHAT_OBSERVATION_BUDGET == 2 * 10000
    assert len(page.records) + len(rest.records) == CHAT_OBSERVATIONS_PER_TURN * 3
    with pytest.raises(ValueError, match="chat stream"):
        api.observations(ObservationsRequest(ObservationStreamRef(ObservationFamily.EVALUATION, PROJECT, session.ref.session_id)))


def test_stale_ready_head_not_owned_here_is_runtime_unavailable_for_turn_and_close():
    records = InMemoryDurableRecordStoreV1()
    first, _, _ = _api({"fake": FakeRuntime()}, records=records)
    session = first.open(_open())
    second, _, _ = _api({"fake": FakeRuntime()}, records=records)
    assert second.show(session.ref).state is ChatSessionState.READY
    for call in (lambda: second.turn(ChatTurnRequest(session.ref, "hi")), lambda: second.close(session.ref)):
        with pytest.raises(ChatOperationError) as captured:
            call()
        assert captured.value.code is ChatOperationCode.RUNTIME_UNAVAILABLE
    assert first.close(session.ref).state is ChatSessionState.CLOSED


def test_tampered_head_is_integrity_error():
    api, records, _ = _api({"fake": FakeRuntime()})
    session = api.open(_open())
    (key,) = _stores(records)[StoragePartition.CHAT_SESSION.value]
    stored = records.read(partition=StoragePartition.CHAT_SESSION.value, key=key)
    document = json.loads(stored.canonical)
    document["record"]["turns"] = 5
    forged = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert records.compare_and_swap(
        partition=StoragePartition.CHAT_SESSION.value, key=key, expected_revision=stored.revision, canonical=forged,
    )
    with pytest.raises(ChatOperationError) as captured:
        api.show(session.ref)
    assert captured.value.code is ChatOperationCode.INTEGRITY_ERROR


# --- composition through compose_reference_host ---------------------------------------------------


class Secrets:
    def resolve(self, reference: SecretRef) -> str:
        assert reference == SECRET
        return "reference-test-master-secret-value-" + "x" * 32


class HostGrants:
    def authorize(self, requirements):
        return ExecutionGrant("host-grant-1")

    def bind(self, grant, *, operation, requirements):
        return object()


class Loader:
    def load(self, canonical_json):
        return TrainingRequest("request", RUN.project_ref, canonical_json)


class Resolver:
    def resolve(self, request):
        return ResolvedTrainingRequest(
            "synaptic-resolved-training-request/v1", request.request_id, request.project_ref, *D[:5],
        )


class Identity:
    def for_plan(self, plan):
        return RUN


def _composed_host(chat):
    records = InMemoryDurableRecordStoreV1()
    streams = RecordingStreams()
    descriptor = ProviderDescriptor(
        "synaptic-provider-descriptor/v1", PROVIDER.provider_id, "Fake provider", "1.0.0",
        ProviderCapabilities(True, True, True, True, True, False),
    )
    basis = TrainingPlanBasisV1("synaptic-training-plan-basis/v1", "request", RUN.project_ref, *D[:5])
    plan_context = ProviderPlanContextV1(
        "synaptic-provider-plan-context/v1", PROVIDER, basis.basis_digest, descriptor.descriptor_digest, D[5],
    )
    executor = ExecutorDescriptorV1(PROVIDER.provider_id, "executor", "1.0.0")
    adapter = AdapterDescriptorV1(PROVIDER.provider_id, "adapter", "1.0.0")
    scope = ExecutionScopeV1("account", "namespace")
    binding = ProviderExecutionBindingV1(
        PROVIDER, descriptor.descriptor_digest, plan_context.profile_digest, scope, executor,
        adapter.digest, D[7], D[8], D[9],
    )
    artifact_bytes = b"adapter-data"
    config = FakeProviderConfigV1(
        PROVIDER, descriptor, plan_context.profile_digest, scope.account_ref, scope.namespace_ref,
        executor, adapter, (), (ProviderRunPhaseV1.RUNNING, ProviderRunPhaseV1.SUCCEEDED),
        (RunLogEntry(1, "2026-08-27T12:00:00Z", RunLogLevel.INFO, "progress", "ok", 2),),
        (FakeArtifactV1(
            VerifiedArtifact("adapter", hashlib.sha256(artifact_bytes).hexdigest(), len(artifact_bytes)),
            artifact_bytes,
        ),),
    )
    ports = ReferenceHostPortsV1(records, streams, Clock(), HostGrants(), Secrets())
    authority = compose_reference_authority(clock=ports.clock, secrets=ports.secrets, authority_secret=SECRET)
    fake = FakeProviderFamilyV1(
        config, evidence_key=b"e" * 32,
        foundation_authenticator=authority.foundation_authenticator,
        assessment_authenticator=authority.assessment_authority,
    )

    class Planning:
        def describe(self, provider):
            return descriptor

        def context(self, resolved, provider):
            return plan_context

        def preflight(self, plan):
            return TrainingPreflight(
                plan.plan_fingerprint, True, "2026-08-26T00:00:00Z", "2026-08-28T00:00:00Z",
                (AuthorizationRequirement("training.start", True, 100, "USD"),),
            )

    class Preparation:
        def resolve(self, provider, plan_context):
            return binding

        def prepare(self, plan, run, binding):
            return CanonicalPreparationV2.build(
                provider=binding.provider, scope=binding.scope, project_ref=run.project_ref,
                run_id=run.run_id, plan_fingerprint=plan.plan_fingerprint,
                source_digest=plan.basis.source_digest, workload_digest=plan.basis.workload_digest,
                runtime_digest=plan.basis.runtime_digest, resource_digest=binding.resource_digest,
                artifact_contract_digest=plan.basis.artifact_policy_digest,
                quote_digest=binding.quote_digest,
                secret_requirements_digest=binding.secret_requirements_digest,
                execution_binding_digest=binding.binding_digest,
            )

        def payload(self, preparation, kind):
            return CanonicalProviderPayloadV1.build(
                PROVIDER.provider_id, f"{kind.value}-payload/v2", preparation.workload_digest,
            )

    family = ProviderFamilyV1(
        descriptor, Planning(), Preparation(), fake.reader, fake.executor_resolver,
        fake.reconciliation_resolver, fake.evidence_authority, fake.artifact_verifier,
        fake.evidence_authority, fake.evidence_authority,
    )
    composition = compose_reference_host(
        family=family, ports=ports, requests=ReferenceRequestPortsV1(Loader(), Resolver(), Identity()),
        authority=authority, chat=chat,
    )
    return composition, composition.api(), records, streams


def test_compose_reference_host_composes_the_chat_family_lazily_and_only_when_asked():
    without, host, records, streams = _composed_host(None)
    assert without.chat is None
    with pytest.raises(RuntimeError, match="did not compose the 'chat' family"):
        host.chat
    runtime = FakeRuntime()
    with_chat, host, records, streams = _composed_host(ReferenceChatPortsV1(ChatRuntimeRegistryV1({"fake": runtime})))
    assert type(host.chat) is ChatAPI
    session = host.chat.open(_open())
    assert host.chat.turn(ChatTurnRequest(session.ref, "hi")).content == "hello there"
    assert host.chat.close(session.ref).state is ChatSessionState.CLOSED
    partitions = set(_stores(records))
    assert StoragePartition.CHAT_SESSION.value in partitions
    assert StoragePartition.WORKFLOW.value not in partitions
    assert {partition for partition, _, _ in streams.appended} == {StoragePartition.OBSERVATION.value}
    with pytest.raises(TypeError, match="exact ReferenceChatPortsV1"):
        _composed_host(object())


def test_chat_ports_and_registry_require_their_methods():
    with pytest.raises(TypeError):
        ReferenceChatPortsV1(object())
    with pytest.raises(TypeError):
        ChatRuntimeRegistryV1({"fake": object()})
    with pytest.raises(TypeError):
        ChatRuntimeRegistryV1({"": FakeRuntime()})
    registry = ChatRuntimeRegistryV1({"fake": FakeRuntime()})
    assert registry.resolve("missing") is None and registry.resolve(None) is None


# --- import closure --------------------------------------------------------------------------------


def test_public_package_import_loads_no_engine_provider_or_database_modules():
    script = f"""
import json, sys
sys.path.insert(0, {str(ROOT)!r})
import synaptic_tuner.api.v1
from synaptic_tuner.api.v1 import ChatAPI, ChatSession, ChatTurn
print(json.dumps(sorted(n for n in sys.modules if n in ('tuner', 'sqlite3', 'modal', 'huggingface_hub', 'runpod') or n.startswith(('tuner.', 'modal.', 'sqlite3.', 'huggingface_hub.', 'runpod.')))))
"""
    completed = subprocess.run([sys.executable, "-I", "-c", script], cwd=ROOT, check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == []
