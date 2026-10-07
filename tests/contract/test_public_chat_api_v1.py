"""Contract tests for the ChatAPI facade (api-facade slice 8, contract only)."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import jsonschema
import pytest

from synaptic_tuner.api import v1
from synaptic_tuner.api.v1 import chat_facade
from synaptic_tuner.api.v1.chat_facade import (
    ChatAPI,
    ChatListRequest,
    ChatModelIdentity,
    ChatModelKind,
    ChatModelSource,
    ChatOpenRequest,
    ChatOperationCode,
    ChatOperationError,
    ChatOperations,
    ChatSession,
    ChatSessionPage,
    ChatSessionPolicyV1,
    ChatSessionRef,
    ChatSessionState,
    ChatTurn,
    ChatTurnRef,
    ChatTurnRequest,
    content_digest,
)
from synaptic_tuner.api.v1.host import APIHost, HostPorts
from synaptic_tuner.api.v1.observations import (
    ChatTurnCompletedPayloadV1,
    ObservationFamily,
    ObservationKind,
    ObservationPage,
    ObservationRecordV1,
    ObservationStreamRef,
    ObservationsRequest,
)
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.usage import SpendRef, UsageAvailability, UsageRecordV1


ROOT = Path(__file__).resolve().parents[2]
SESSION_SCHEMA = json.loads((ROOT / "schemas/synaptic-chat-session-v1.schema.json").read_text(encoding="utf-8"))
TURN_SCHEMA = json.loads((ROOT / "schemas/synaptic-chat-turn-v1.schema.json").read_text(encoding="utf-8"))

VERBS = {"open", "turn", "show", "list", "close", "observations"}
SESSION = ChatSessionRef("cs-01J", "acme")
RUN = TrainingRunRef("run-7", "acme")
HUB = ChatModelSource("acme/qwen-sft", "a1b2" * 10)
IDENTITY = ChatModelIdentity("acme/qwen-sft", "a1b2" * 10, "a1b2" * 10, ChatModelKind.FULL)


def _policy(**changes: object) -> ChatSessionPolicyV1:
    values: dict[str, object] = {
        "request_timeout_seconds": 30.0, "idle_timeout_seconds": 600.0,
        "absolute_lifetime_seconds": 3600.0, "max_turns": 50, "max_history_bytes": 1 << 20,
    }
    values.update(changes)
    return ChatSessionPolicyV1(**values)  # type: ignore[arg-type]


def _open(**changes: object) -> ChatOpenRequest:
    values: dict[str, object] = {"project_ref": "acme", "runtime": "local", "policy": _policy(), "model": HUB}
    values.update(changes)
    return ChatOpenRequest(**values)  # type: ignore[arg-type]


def _session(**changes: object) -> ChatSession:
    values: dict[str, object] = {
        "schema_version": "synaptic-chat-session/v1", "ref": SESSION, "state": ChatSessionState.READY,
        "model": IDENTITY, "run": None, "turns": 3, "history_bytes": 4096, "policy": _policy(),
        "diagnostic_code": None,
    }
    values.update(changes)
    return ChatSession(**values)  # type: ignore[arg-type]


def _measured_usage() -> UsageRecordV1:
    return UsageRecordV1("synaptic-usage/v1", UsageAvailability.MEASURED, 412, 88, None, None, None)


def _turn(content: str = "The answer is 42.\nSecond line.", **changes: object) -> ChatTurn:
    values: dict[str, object] = {
        "schema_version": "synaptic-chat-turn/v1", "ref": ChatTurnRef(SESSION, 4), "state": ChatSessionState.READY,
        "content_digest": content_digest(content), "content": content, "usage": None,
    }
    values.update(changes)
    return ChatTurn(**values)  # type: ignore[arg-type]


def _stream() -> ObservationStreamRef:
    return ObservationStreamRef(ObservationFamily.CHAT, "acme", "cs-01J")


def _observation(sequence: int = 8) -> ObservationRecordV1:
    return ObservationRecordV1(
        "synaptic-observation/v1", _stream(), sequence, "2026-09-17T12:00:04Z",
        ObservationKind.CHAT_TURN_COMPLETED, ChatTurnCompletedPayloadV1(4, "c" * 64, 17),
    )


# --- exports, verbs, closed vocabularies ---------------------------------------


def test_root_chat_exports_are_the_canonical_facade_identities() -> None:
    for name in chat_facade.__all__:
        assert getattr(v1, name) is getattr(chat_facade, name), name
    assert set(chat_facade.__all__) <= set(v1.__all__)


def test_chat_operations_protocol_and_facade_expose_exactly_the_six_verbs() -> None:
    protocol = {name for name, member in vars(ChatOperations).items() if inspect.isfunction(member) and not name.startswith("_")}
    assert protocol == VERBS
    facade = {name for name, member in vars(ChatAPI).items() if inspect.isfunction(member) and not name.startswith("_")}
    assert facade == VERBS
    for verb in VERBS:
        assert inspect.signature(getattr(ChatOperations, verb)).parameters.keys() == inspect.signature(getattr(ChatAPI, verb)).parameters.keys()


def test_closed_vocabularies_are_exact() -> None:
    assert tuple(state.value for state in ChatSessionState) == (
        "opening", "ready", "serving", "closing", "closed", "cleanup_unresolved",
    )
    assert tuple(kind.value for kind in ChatModelKind) == ("full", "lora")
    assert tuple(code.value for code in ChatOperationCode) == (
        "session_missing", "session_closed", "session_busy", "turn_bounds_invalid", "model_ineligible",
        "runtime_unavailable", "cleanup_unresolved", "host_unsupported", "integrity_error",
    )
    assert "cursor_invalid" not in {code.value for code in ChatOperationCode}
    for enum_type in (ChatSessionState, ChatModelKind, ChatOperationCode):
        assert issubclass(enum_type, str)
        with pytest.raises(ValueError):
            enum_type("streaming")


def test_operation_error_requires_the_exact_code() -> None:
    error = ChatOperationError(ChatOperationCode.SESSION_BUSY)
    assert error.code is ChatOperationCode.SESSION_BUSY
    assert str(error) == "session_busy"
    assert isinstance(error, ValueError)
    with pytest.raises(TypeError, match="exact ChatOperationCode"):
        ChatOperationError("session_busy")  # type: ignore[arg-type]


# --- record shapes and schemas ------------------------------------------------------


def test_chat_session_record_has_exact_fields_and_round_trips_through_its_schema() -> None:
    record = _session(run=RUN)
    document = record.to_dict()
    assert tuple(document) == (
        "schema_version", "ref", "state", "model", "run", "turns", "history_bytes", "policy", "diagnostic_code",
    )
    jsonschema.Draft202012Validator.check_schema(SESSION_SCHEMA)
    jsonschema.validate(document, SESSION_SCHEMA)
    assert ChatSession.from_dict(json.loads(json.dumps(document))) == record
    with pytest.raises(ValueError, match="unknown fields"):
        ChatSession.from_dict(dict(document, streaming=True))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(document, streaming=True), SESSION_SCHEMA)
    opening = _session(state=ChatSessionState.OPENING, model=None, turns=0, history_bytes=0)
    jsonschema.validate(opening.to_dict(), SESSION_SCHEMA)
    assert ChatSession.from_dict(opening.to_dict()) == opening
    never_ready = _session(
        state=ChatSessionState.CLEANUP_UNRESOLVED, model=None, turns=0, history_bytes=0,
        diagnostic_code="cleanup_unresolved",
    )
    jsonschema.validate(never_ready.to_dict(), SESSION_SCHEMA)
    assert ChatSession.from_dict(never_ready.to_dict()) == never_ready


@pytest.mark.parametrize(
    "changes",
    [
        {"state": ChatSessionState.READY, "diagnostic_code": "closed_early"},
        {"state": ChatSessionState.CLEANUP_UNRESOLVED},
        {"model": None},
        {"model": None, "state": ChatSessionState.OPENING},
        {"turns": 10001},
        {"history_bytes": (64 << 20) + 1},
        {"schema_version": "synaptic-chat-session/v2"},
    ],
)
def test_chat_session_invariants_are_enforced_by_record_and_schema(changes) -> None:
    with pytest.raises((TypeError, ValueError)):
        _session(**changes)
    document = _session().to_dict()
    for name, value in changes.items():
        document[name] = value.to_dict() if hasattr(value, "to_dict") else (value.value if hasattr(value, "value") else value)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(document, SESSION_SCHEMA)


@pytest.mark.parametrize(
    "changes",
    [
        {"turns": 51},
        {"history_bytes": (1 << 20) + 1},
        {"run": TrainingRunRef("run-7", "other")},
    ],
)
def test_chat_session_cross_field_invariants_are_enforced_by_the_record(changes) -> None:
    """Bounds against the session's own policy and the run's project are record invariants the schema cannot express."""
    with pytest.raises(ValueError):
        _session(**changes)
    with pytest.raises(ValueError):
        ChatSession.from_dict(_session(policy=_policy(max_turns=100, max_history_bytes=2 << 20)).to_dict() | {
            name: (value.to_dict() if hasattr(value, "to_dict") else value) for name, value in changes.items()
        } | {"policy": _policy().to_dict()})


def test_chat_turn_round_trips_and_omits_unavailable_usage() -> None:
    turn = _turn()
    document = turn.to_dict()
    assert tuple(document) == ("schema_version", "ref", "state", "content_digest", "content")
    assert "usage" not in document
    jsonschema.Draft202012Validator.check_schema(TURN_SCHEMA)
    jsonschema.validate(document, TURN_SCHEMA)
    assert ChatTurn.from_dict(json.loads(json.dumps(document))) == turn
    measured = _turn(usage=_measured_usage())
    measured_document = measured.to_dict()
    assert tuple(measured_document) == ("schema_version", "ref", "state", "content_digest", "content", "usage")
    jsonschema.validate(measured_document, TURN_SCHEMA)
    assert ChatTurn.from_dict(measured_document) == measured
    with pytest.raises(ValueError, match="omitted"):
        ChatTurn.from_dict(dict(document, usage=None))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(document, usage=None), TURN_SCHEMA)
    unavailable = UsageRecordV1("synaptic-usage/v1", UsageAvailability.UNAVAILABLE, None, None, None, None, None)
    with pytest.raises(ValueError, match="omitted"):
        _turn(usage=unavailable)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(document, usage=unavailable.to_dict()), TURN_SCHEMA)
    spend = UsageRecordV1(
        "synaptic-usage/v1", UsageAvailability.MEASURED, 1, 2, 3, "USD", SpendRef("modal", "acct", "spend-1"),
    )
    jsonschema.validate(_turn(usage=spend).to_dict(), TURN_SCHEMA)


@pytest.mark.parametrize(
    "changes",
    [
        {"state": ChatSessionState.OPENING},
        {"state": ChatSessionState.SERVING},
        {"content_digest": "d" * 64},
        {"content_digest": "D" * 64},
        {"schema_version": "synaptic-chat-turn/v2"},
    ],
)
def test_chat_turn_invariants_are_enforced_by_record_and_schema(changes) -> None:
    with pytest.raises((TypeError, ValueError)):
        _turn(**changes)
    document = _turn().to_dict()
    for name, value in changes.items():
        document[name] = value.value if hasattr(value, "value") else value
    if "content_digest" in changes and changes["content_digest"] == "d" * 64:
        jsonschema.validate(document, TURN_SCHEMA)  # the schema cannot hash; the record does
        return
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(document, TURN_SCHEMA)


def test_turn_content_is_plain_text_and_the_digest_binds_it() -> None:
    content = "line one\n\tindented line two\nend"
    turn = _turn(content, content_digest=content_digest(content))
    assert turn.content == content
    assert turn.content_digest == content_digest(content)
    with pytest.raises(ValueError, match="does not match"):
        _turn(content, content_digest=content_digest(content + " "))


# --- identities: request_id consecutiveness, model identity, sources ----------------------


def test_turn_ref_request_id_is_a_positive_consecutive_integer_from_one() -> None:
    first = ChatTurnRef(SESSION, 1)
    second = ChatTurnRef(SESSION, 2)
    assert first.follows(None) and second.follows(first)
    assert not second.follows(None)
    assert not ChatTurnRef(SESSION, 3).follows(first)
    assert not ChatTurnRef(ChatSessionRef("cs-other", "acme"), 2).follows(first)
    assert ChatTurnRef.from_dict(second.to_dict()) == second
    for bad in (0, -1, 10001):
        with pytest.raises(ValueError, match="request_id"):
            ChatTurnRef(SESSION, bad)
    for bad in (True, 1.0, "1"):
        with pytest.raises(TypeError, match="request_id"):
            ChatTurnRef(SESSION, bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact ChatTurnRef"):
        second.follows(object())  # type: ignore[arg-type]


def test_model_identity_pins_revisions_and_closed_kind() -> None:
    assert ChatModelIdentity.from_dict(IDENTITY.to_dict()) == IDENTITY
    local = ChatModelIdentity("/srv/models/qwen", "f" * 64, "f" * 64, ChatModelKind.LORA)
    assert ChatModelIdentity.from_dict(local.to_dict()) == local
    for bad in ("main", "A1B2" * 10, "a" * 39, "a" * 41, "a" * 65):
        with pytest.raises(ValueError, match="revision"):
            ChatModelIdentity("acme/qwen-sft", bad, "a" * 40, ChatModelKind.FULL)
    with pytest.raises(TypeError, match="exact ChatModelKind"):
        ChatModelIdentity("acme/qwen-sft", "a" * 40, "a" * 40, "full")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown model_kind"):
        ChatModelIdentity.from_dict(dict(IDENTITY.to_dict(), model_kind="merged"))


def test_model_source_is_a_canonical_local_directory_or_a_pinned_hub_ref() -> None:
    assert HUB.identity() == IDENTITY
    assert ChatModelSource.from_dict(HUB.to_dict()) == HUB
    lora = ChatModelSource("/srv/models/base", "e" * 64, "d" * 40, ChatModelKind.LORA, "/srv/adapters/sft")
    assert lora.identity() == ChatModelIdentity("/srv/models/base", "e" * 64, "d" * 40, ChatModelKind.LORA)
    assert ChatModelSource.from_dict(lora.to_dict()) == lora
    with pytest.raises(ValueError, match="revision"):
        ChatModelSource("acme/qwen-sft", "main")
    with pytest.raises(ValueError, match="40-hex commit"):
        ChatModelSource("acme/qwen-sft", "e" * 64)
    with pytest.raises(ValueError, match="64-hex"):
        ChatModelSource("/srv/models/base", "a" * 40)
    with pytest.raises(ValueError, match="absolute directory or an owner/name"):
        ChatModelSource("relative/../path/model", "a" * 40)
    with pytest.raises(ValueError, match="absolute directory or an owner/name"):
        ChatModelSource("qwen-sft", "a" * 40)
    with pytest.raises(ValueError, match="exactly one adapter_path"):
        ChatModelSource("acme/qwen-sft", "a" * 40, model_kind=ChatModelKind.LORA)
    with pytest.raises(ValueError, match="exactly one adapter_path"):
        ChatModelSource("acme/qwen-sft", "a" * 40, adapter_path="/srv/adapters/sft")
    with pytest.raises(ValueError, match="absolute local directory"):
        ChatModelSource("acme/qwen-sft", "a" * 40, model_kind=ChatModelKind.LORA, adapter_path="adapters/sft")


# --- open request and policy bounds ------------------------------------------------------


def test_open_request_selects_exactly_one_of_model_or_run_and_a_named_runtime() -> None:
    model_first = _open()
    assert ChatOpenRequest.from_dict(model_first.to_dict()) == model_first
    run_first = _open(model=None, run=RUN)
    assert ChatOpenRequest.from_dict(run_first.to_dict()) == run_first
    assert tuple(run_first.to_dict()) == ("project_ref", "runtime", "policy", "model", "run")
    with pytest.raises(ValueError, match="exactly one of model or run"):
        _open(run=RUN)
    with pytest.raises(ValueError, match="exactly one of model or run"):
        _open(model=None)
    with pytest.raises(ValueError, match="run project does not match"):
        _open(model=None, run=TrainingRunRef("run-7", "other"))
    with pytest.raises(ValueError):
        _open(runtime="")
    with pytest.raises(TypeError, match="exact ChatSessionPolicyV1"):
        _open(policy=object())
    with pytest.raises(ValueError, match="unknown fields"):
        ChatOpenRequest.from_dict(dict(model_first.to_dict(), gpu="H100"))


@pytest.mark.parametrize(
    "changes",
    [
        {"request_timeout_seconds": 0}, {"request_timeout_seconds": 86401}, {"idle_timeout_seconds": -1.0},
        {"absolute_lifetime_seconds": float("inf")}, {"absolute_lifetime_seconds": float("nan")},
        {"max_turns": 0}, {"max_turns": 10001}, {"max_history_bytes": 0}, {"max_history_bytes": (64 << 20) + 1},
    ],
)
def test_policy_outside_its_bounds_is_turn_bounds_invalid(changes) -> None:
    with pytest.raises(ChatOperationError) as captured:
        _policy(**changes)
    assert captured.value.code is ChatOperationCode.TURN_BOUNDS_INVALID


def test_policy_at_its_bounds_round_trips_and_wrong_types_are_type_errors() -> None:
    edge = _policy(
        request_timeout_seconds=86400, idle_timeout_seconds=0.001, absolute_lifetime_seconds=86400.0,
        max_turns=10000, max_history_bytes=64 << 20,
    )
    assert ChatSessionPolicyV1.from_dict(edge.to_dict()) == edge
    with pytest.raises(TypeError):
        _policy(max_turns=1.0)
    with pytest.raises(TypeError):
        _policy(max_turns=True)
    with pytest.raises(TypeError):
        _policy(request_timeout_seconds="30")


def test_turn_request_is_bounded_and_exact() -> None:
    request = ChatTurnRequest(SESSION, "hello\nthere")
    assert ChatTurnRequest.from_dict(request.to_dict()) == request
    with pytest.raises(ChatOperationError) as captured:
        ChatTurnRequest(SESSION, "")
    assert captured.value.code is ChatOperationCode.TURN_BOUNDS_INVALID
    with pytest.raises(ChatOperationError) as captured:
        ChatTurnRequest(SESSION, "x" * ((64 << 20) + 1))
    assert captured.value.code is ChatOperationCode.TURN_BOUNDS_INVALID
    with pytest.raises(TypeError, match="exact string"):
        ChatTurnRequest(SESSION, b"hello")  # type: ignore[arg-type]


# --- paging ---------------------------------------------------------------------------


def test_chat_session_page_follows_the_settled_paging_rule() -> None:
    request = ChatListRequest("acme", limit=2)
    sessions = (_session(), _session(ref=ChatSessionRef("cs-02", "acme")))
    page = ChatSessionPage(request, sessions, "cs-02", True)
    assert page.truncated and page.next_cursor == "cs-02"
    assert ChatSessionPage(request, sessions[:1]).next_cursor is None
    with pytest.raises(ValueError, match="matrix"):
        ChatSessionPage(request, sessions, None, True)
    with pytest.raises(ValueError, match="matrix"):
        ChatSessionPage(request, sessions, "cs-02", False)
    with pytest.raises(ValueError, match="must contain a session"):
        ChatSessionPage(request, (), "cs-02", True)
    with pytest.raises(ValueError, match="exceed"):
        ChatSessionPage(ChatListRequest("acme", limit=1), sessions)
    with pytest.raises(ValueError, match="project does not match"):
        ChatSessionPage(ChatListRequest("other", limit=2), sessions)
    with pytest.raises(ValueError, match="ASCII"):
        ChatSessionPage(request, sessions, "cs-é", True)
    with pytest.raises(ValueError, match="256 bytes"):
        ChatSessionPage(request, sessions, "c" * 257, True)
    for bad in (0, 101, 1.0):
        with pytest.raises((TypeError, ValueError)):
            ChatListRequest("acme", limit=bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact tuple"):
        ChatSessionPage(request, list(sessions))  # type: ignore[arg-type]


# --- facade discipline ----------------------------------------------------------------------


def test_facade_presents_detached_inputs_and_returns_rebuilt_results() -> None:
    seen = {}

    class Operations:
        def open(self, request):
            seen["open"] = request
            return _session(ref=ChatSessionRef("cs-new", "acme"), turns=0, history_bytes=0)

        def turn(self, request):
            seen["turn"] = request
            return _turn()

        def show(self, session):
            seen["show"] = session
            return _session()

        def list(self, request):
            seen["list"] = request
            return ChatSessionPage(request, (_session(),))

        def close(self, session):
            seen["close"] = session
            return _session(state=ChatSessionState.CLOSED)

        def observations(self, request):
            seen["observations"] = request
            return ObservationPage(request, (_observation(),))

    api = ChatAPI(Operations())
    open_request = _open()
    opened = api.open(open_request)
    assert seen["open"] == open_request and seen["open"] is not open_request
    assert opened.model == IDENTITY and opened.state is ChatSessionState.READY
    turn_request = ChatTurnRequest(SESSION, "hello")
    turn = api.turn(turn_request)
    assert seen["turn"] == turn_request and seen["turn"] is not turn_request
    assert turn == _turn() and turn.ref.session == SESSION
    assert api.show(SESSION) == _session() and seen["show"] is not SESSION
    listing = ChatListRequest("acme", limit=5)
    page = api.list(listing)
    assert page.request == listing and page.sessions == (_session(),)
    assert api.close(SESSION).state is ChatSessionState.CLOSED
    observations = ObservationsRequest(_stream(), limit=10)
    result = api.observations(observations)
    assert result.records == (_observation(),) and seen["observations"] is not observations


@pytest.mark.parametrize(
    "verb, callback, argument",
    [
        ("open", lambda supplied: _session(ref=ChatSessionRef("cs-x", "other")), _open()),
        ("open", lambda supplied: _session(policy=_policy(max_turns=7)), _open()),
        ("open", lambda supplied: _session(state=ChatSessionState.CLOSED, diagnostic_code="x"), _open()),
        ("open", lambda supplied: _session(model=ChatModelIdentity("acme/other", "b" * 40, "b" * 40, ChatModelKind.FULL)), _open()),
        ("open", lambda supplied: _session(run=None), _open(model=None, run=RUN)),
        ("turn", lambda supplied: _turn(ref=ChatTurnRef(ChatSessionRef("cs-x", "acme"), 4)), ChatTurnRequest(SESSION, "hi")),
        ("show", lambda supplied: _session(ref=ChatSessionRef("cs-x", "acme")), SESSION),
        ("close", lambda supplied: _session(ref=ChatSessionRef("cs-x", "acme")), SESSION),
        ("list", lambda supplied: ChatSessionPage(ChatListRequest("acme", limit=2), ()), ChatListRequest("acme", limit=1)),
        ("observations", lambda supplied: ObservationPage(ObservationsRequest(_stream(), limit=2), ()), ObservationsRequest(_stream(), limit=1)),
    ],
)
def test_chat_facade_rejects_callback_identity_drift(verb, callback, argument) -> None:
    operations = type("Operations", (), {verb: staticmethod(callback)})()
    with pytest.raises(ValueError, match="bind"):
        getattr(ChatAPI(operations), verb)(argument)


@pytest.mark.parametrize("verb", sorted(VERBS))
@pytest.mark.parametrize("raises", [False, True])
def test_chat_facade_rejects_presented_input_mutation_on_return_and_raise(verb, raises) -> None:
    open_request = _open()
    turn_request = ChatTurnRequest(SESSION, "hello")
    listing = ChatListRequest("acme", limit=1)
    observations = ObservationsRequest(_stream(), limit=1)

    class Operations:
        def __getattr__(self, name):
            def callback(value, *extra):
                if name in {"open", "list"}:
                    object.__setattr__(value, "project_ref", "changed")
                elif name == "turn":
                    object.__setattr__(value.session, "session_id", "changed")
                elif name == "observations":
                    object.__setattr__(value.stream, "entity_id", "changed")
                else:
                    object.__setattr__(value, "session_id", "changed")
                if raises:
                    raise RuntimeError("collaborator detail")
                return object()
            return callback

    api = ChatAPI(Operations())
    invocation = {
        "open": lambda: api.open(open_request),
        "turn": lambda: api.turn(turn_request),
        "show": lambda: api.show(SESSION),
        "list": lambda: api.list(listing),
        "close": lambda: api.close(SESSION),
        "observations": lambda: api.observations(observations),
    }[verb]
    with pytest.raises(ValueError, match="input changed") as captured:
        invocation()
    if raises:
        pending = [captured.value]
        seen = set()
        while pending:
            error = pending.pop()
            if id(error) in seen:
                continue
            seen.add(id(error))
            assert type(error) is not RuntimeError
            assert "collaborator detail" not in str(error)
            pending.extend(item for item in (error.__cause__, error.__context__) if item is not None)


def test_observations_verb_admits_only_chat_streams_before_any_callback() -> None:
    calls = []

    class Operations:
        def observations(self, supplied):
            calls.append(supplied)
            return ObservationPage(supplied, ())

    api = ChatAPI(Operations())
    foreign = ObservationsRequest(ObservationStreamRef(ObservationFamily.EVALUATION, "acme", "ev-1"))
    with pytest.raises(ValueError, match="chat stream"):
        api.observations(foreign)
    assert calls == []
    assert api.observations(ObservationsRequest(_stream())).records == ()


def test_facade_surfaces_operation_errors_unchanged() -> None:
    class Operations:
        def turn(self, request):
            raise ChatOperationError(ChatOperationCode.SESSION_BUSY)

    with pytest.raises(ChatOperationError) as captured:
        ChatAPI(Operations()).turn(ChatTurnRequest(SESSION, "hi"))
    assert captured.value.code is ChatOperationCode.SESSION_BUSY


# --- hostile inputs -----------------------------------------------------------------


def test_canonical_chat_contracts_reject_nonexact_inputs() -> None:
    class Text(str):
        pass

    class SessionSubclass(ChatSessionRef):
        pass

    class DictSubclass(dict):
        pass

    with pytest.raises(TypeError):
        ChatSessionRef(Text("cs-01J"), "acme")
    with pytest.raises(TypeError, match="exact object"):
        ChatSessionRef.from_dict(MappingProxyType({"session_id": "cs-01J", "project_ref": "acme"}))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact object"):
        ChatSession.from_dict(DictSubclass(_session().to_dict()))
    with pytest.raises(TypeError, match="exact object"):
        ChatTurn.from_dict(DictSubclass(_turn().to_dict()))
    with pytest.raises(TypeError, match="exact ChatSessionRef"):
        ChatAPI(object()).show(SessionSubclass("cs-01J", "acme"))
    with pytest.raises(TypeError, match="exact ChatSessionRef"):
        ChatTurnRef(SessionSubclass("cs-01J", "acme"), 1)
    with pytest.raises(TypeError, match="exact integer"):
        _session(turns=3.0)
    with pytest.raises(TypeError, match="field names"):
        ChatTurn.from_dict({Text("schema_version"): "synaptic-chat-turn/v1"})  # type: ignore[dict-item]


# --- host composition and import closure --------------------------------------------


def test_api_host_composes_the_chat_facade() -> None:
    class Clock:
        def now(self):
            return "2026-09-17T12:00:00Z"

    class Operations:
        def show(self, session):
            return _session()

    ports = HostPorts(
        training=object(), runs=object(), artifacts=None, evaluation=None,
        chat=Operations(), data=None, pipelines=None, clock=Clock(),
    )
    host = APIHost(ports)
    assert type(host.chat) is ChatAPI
    assert host.chat.show(SESSION) == _session()
    with pytest.raises(RuntimeError, match="did not compose the 'chat' family"):
        APIHost(HostPorts(
            training=object(), runs=object(), artifacts=None, evaluation=None,
            chat=None, data=None, pipelines=None, clock=Clock(),
        )).chat


def test_chat_facade_imports_no_engine_provider_or_database_modules() -> None:
    script = f"""
import json, sys
sys.path.insert(0, {str(ROOT)!r})
import synaptic_tuner.api.v1.chat_facade
print(json.dumps(sorted(n for n in sys.modules if n in ('tuner', 'sqlite3', 'modal', 'huggingface_hub', 'runpod', 'Evaluator', 'SynthChat') or n.startswith(('tuner.', 'modal.', 'sqlite3.', 'huggingface_hub.', 'runpod.', 'Evaluator.', 'SynthChat.')))))
"""
    completed = subprocess.run([sys.executable, "-I", "-c", script], cwd=ROOT, check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == []
