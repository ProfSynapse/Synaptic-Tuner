"""Host-side acceptance tests for one owned Modal chat channel."""

from __future__ import annotations

import hashlib
import json
from queue import Queue
import threading
import time
from types import MappingProxyType

import pytest

from Evaluator.chat_session import ChatSession, ChatSessionPolicy
from Evaluator.protocols import BackendResponse
from tuner.inference.run_chat import PreparedModelIdentity
from tuner.execution.providers.modal.inference_channel import (
    decode_modal_chat_frame,
    encode_modal_chat_frame,
    serve_modal_chat_channel,
)
from tuner.execution.providers.modal.inference_channel_client import (
    ModalInferenceChannelClient,
    ModalInferenceChannelClientError,
)

ARGUMENT = b"authenticated-launch"
SESSION = "chat-session-1"
LAUNCH = hashlib.sha256(ARGUMENT).hexdigest()
MODEL = PreparedModelIdentity("example/model", "a" * 40, "b" * 40, "lora")
MODEL_DICT = {
    "model_ref": MODEL.model_ref,
    "model_revision": MODEL.model_revision,
    "tokenizer_revision": MODEL.tokenizer_revision,
    "model_kind": MODEL.model_kind,
}


def _frame(kind, **values):
    if kind == "ready" and "model" not in values:
        values["model"] = dict(MODEL_DICT)
    return {
        "schema_version": "synaptic-modal-chat-channel/v1",
        "kind": kind,
        "session_id": SESSION,
        "launch_digest": LAUNCH,
        **values,
    }


def _line(kind, **values):
    return encode_modal_chat_frame(_frame(kind, **values), 4096).decode("utf-8")


class Input:
    def __init__(self):
        self.writes = []
        self.drains = 0
        self.block = None
        self.drain_error = None

    def write(self, value):
        self.writes.append(value)

    def drain(self):
        self.drains += 1
        if self.drain_error is not None:
            raise self.drain_error
        if self.block is not None:
            self.block.wait()


class Output:
    def __init__(self, *values):
        self.values = Queue()
        self.reads = 0
        for value in values:
            self.values.put(value)

    def __iter__(self):
        return self

    def __next__(self):
        self.reads += 1
        value = self.values.get()
        if isinstance(value, BaseException):
            raise value
        if value is StopIteration:
            raise StopIteration
        return value


class Sandbox:
    def __init__(self, stdout):
        self._object_id = "sb-fixture"
        self._stdin = Input()
        self._stdout = stdout
        self.terminations = 0
        self.terminate_gate = None
        self.terminate_result = 0

    @property
    def object_id(self):
        return self._object_id

    @property
    def stdin(self):
        return self._stdin

    @property
    def stdout(self):
        return self._stdout

    def terminate(self, *, wait=False):
        self.terminations += 1
        assert wait is True
        if self.terminate_gate is not None:
            self.terminate_gate.wait()
        return self.terminate_result


def _client(sandbox, **changes):
    kwargs = dict(
        session_id=SESSION,
        argument_bytes=ARGUMENT,
        deadline=time.monotonic() + 10,
        startup_timeout_seconds=1,
        request_timeout_seconds=1,
        max_request_bytes=4096,
        max_response_bytes=4096,
    )
    kwargs.update(changes)
    return ModalInferenceChannelClient(sandbox, **kwargs)


def _messages(*rows):
    return tuple(
        MappingProxyType({"role": role, "content": content}) for role, content in rows
    )


def test_ready_and_sequential_chat_use_exact_codec_and_prior_prefix():
    output = Output(
        _line("ready"),
        _line("chat", request_id=1, content="first-answer"),
        _line("chat", request_id=2, content="second-answer"),
    )
    sandbox = Sandbox(output)
    client = _client(sandbox)
    assert client.model == MODEL
    assert client.model is not client.model
    first = client.chat(_messages(("user", "first")))
    second = client.chat(
        _messages(
            ("user", "first"),
            ("assistant", "first-answer"),
            ("user", "second"),
        )
    )
    assert type(first) is type(second) is BackendResponse
    assert (first.message, second.message) == ("first-answer", "second-answer")
    assert first.raw == _frame("chat", request_id=1, content="first-answer")
    sent = [decode_modal_chat_frame(value, 4096) for value in sandbox.stdin.writes]
    assert sent == [
        _frame("chat", request_id=1, content="first"),
        _frame("chat", request_id=2, content="second"),
    ]
    assert sandbox.stdin.drains == 2


@pytest.mark.parametrize(
    "messages",
    (
        [],
        ({"role": "user", "content": "first", "extra": "value"},),
        _messages(("assistant", "first")),
        _messages(("user", "")),
        _messages(("user", "first"), ("user", "extra")),
    ),
)
def test_invalid_or_nonexact_history_is_denied_without_write(messages):
    sandbox = Sandbox(Output(_line("ready")))
    client = _client(sandbox)
    with pytest.raises(ModalInferenceChannelClientError):
        client.chat(messages)
    assert sandbox.stdin.writes == []


@pytest.mark.parametrize(
    "response",
    (
        _line("chat", request_id=2, content="wrong-id"),
        encode_modal_chat_frame(
            _frame("chat", request_id=1, content="wrong-launch")
            | {"launch_digest": "0" * 64},
            4096,
        ).decode(),
        _line("error", request_id=1, code="modal_chat_channel_error"),
        _line("closed", request_id=1),
        _line("closed", request_id=2),
    ),
)
def test_wrong_bound_error_or_async_closed_response_poison_without_retry(response):
    sandbox = Sandbox(Output(_line("ready"), response))
    client = _client(sandbox)
    with pytest.raises(ModalInferenceChannelClientError):
        client.chat(_messages(("user", "first")))
    assert len(sandbox.stdin.writes) == 1
    with pytest.raises(ModalInferenceChannelClientError):
        client.chat(_messages(("user", "again")))
    assert len(sandbox.stdin.writes) == 1


def test_oversized_request_is_frozen_and_denied_before_write():
    sandbox = Sandbox(Output(_line("ready")))
    client = _client(sandbox, max_request_bytes=256)
    with pytest.raises(ModalInferenceChannelClientError) as caught:
        client.chat(_messages(("user", "x" * 500)))
    assert str(caught.value) == "modal_inference_channel_client_invalid"
    assert caught.value.__cause__ is None
    assert sandbox.stdin.writes == []


def test_eof_and_oversized_output_are_closed_without_second_read():
    for response in (
        StopIteration,
        "x" * 4097,
        _line("chat", request_id=1, content="x").encode(),
    ):
        sandbox = Sandbox(Output(_line("ready"), response))
        client = _client(sandbox)
        with pytest.raises(ModalInferenceChannelClientError):
            client.chat(_messages(("user", "first")))
        assert len(sandbox.stdin.writes) == 1


@pytest.mark.parametrize(
    "change",
    (
        {"deadline": True},
        {"deadline": float("nan")},
        {"deadline": "beyond_current_limit"},
        {"startup_timeout_seconds": 0},
        {"startup_timeout_seconds": 1e308},
        {"request_timeout_seconds": -1},
        {"request_timeout_seconds": 1e308},
        {"max_request_bytes": True},
        {"max_response_bytes": 1024 * 1024 + 1},
    ),
)
def test_invalid_constructor_bounds_fail_closed_and_terminate(change):
    sandbox = Sandbox(Output(_line("ready")))
    if change.get("deadline") == "beyond_current_limit":
        # Compute at invocation, not test collection: a long earlier test
        # must not turn an over-budget deadline into an admissible one.
        change = {"deadline": time.monotonic() + 24 * 60 * 60 + 1}
    with pytest.raises(ModalInferenceChannelClientError) as caught:
        _client(sandbox, **change)
    assert str(caught.value) == "modal_inference_channel_client_invalid"
    assert caught.value.__cause__ is None
    limit = time.monotonic() + 1
    while sandbox.terminations == 0 and time.monotonic() < limit:
        time.sleep(0.001)
    assert sandbox.terminations == 1
    assert sandbox.stdout.reads == 0


@pytest.mark.parametrize("name", ("term_timeout", "kill_timeout"))
def test_close_rejects_unbounded_wait_before_stop_or_termination(name):
    sandbox = Sandbox(Output(_line("ready")))
    client = _client(sandbox)
    with pytest.raises(ValueError):
        client.close(**{name: 1e308})
    assert sandbox.stdin.writes == []
    assert sandbox.terminations == 0
    assert client.close(term_timeout=0, kill_timeout=1) is True


def test_close_sends_one_stop_accepts_closed_and_terminates_exact_owner_once():
    sandbox = Sandbox(Output(_line("ready"), _line("closed", request_id=1)))
    client = _client(sandbox)
    assert client.close(term_timeout=1, kill_timeout=1) is True
    assert client.cleanup_pending is False
    assert sandbox.terminations == 1
    assert [decode_modal_chat_frame(value, 4096) for value in sandbox.stdin.writes] == [
        _frame("stop", request_id=1)
    ]
    assert client.close(term_timeout=1, kill_timeout=1) is True
    assert sandbox.terminations == 1


def test_hung_termination_retains_owner_and_later_close_recovers_without_retry():
    gate = threading.Event()
    sandbox = Sandbox(Output(_line("ready"), _line("closed", request_id=1)))
    sandbox.terminate_gate = gate
    client = _client(sandbox)
    assert client.close(term_timeout=1, kill_timeout=0.001) is False
    assert client.cleanup_pending is True
    assert sandbox.terminations == 1
    gate.set()
    assert client.close(term_timeout=0, kill_timeout=1) is True
    assert sandbox.terminations == 1


def test_concurrent_chat_is_rejected_and_close_terminates_without_stop():
    gate = threading.Event()
    sandbox = Sandbox(
        Output(_line("ready"), _line("chat", request_id=1, content="late"))
    )
    sandbox.stdin.block = gate
    client = _client(sandbox)
    failures = []

    def chat():
        try:
            client.chat(_messages(("user", "first")))
        except BaseException as error:
            failures.append(error)

    worker = threading.Thread(target=chat)
    worker.start()
    while not sandbox.stdin.writes:
        time.sleep(0.001)
    with pytest.raises(ModalInferenceChannelClientError):
        client.chat(_messages(("user", "second")))
    assert client.close(term_timeout=0.01, kill_timeout=1) is True
    gate.set()
    worker.join(1)
    assert not worker.is_alive()
    assert len(sandbox.stdin.writes) == 1
    assert len(failures) == 1
    assert sandbox.terminations == 1


def test_bad_ready_or_startup_timeout_terminates_and_retains_pending_owner():
    for output in (
        Output(_line("ready")[:-1] + "x\n"),
        Output(),
    ):
        sandbox = Sandbox(output)
        with pytest.raises(ModalInferenceChannelClientError) as caught:
            _client(sandbox, startup_timeout_seconds=0.001)
        assert str(caught.value) == "modal_inference_channel_client_invalid"
        lease = getattr(caught.value, "cleanup_lease", None)
        assert type(lease) is ModalInferenceChannelClient
        assert lease._sandbox is sandbox
        assert lease.close(term_timeout=0, kill_timeout=1) is True
        assert sandbox.terminations == 1


def test_constructor_failure_exposes_exact_pending_cleanup_lease():
    gate = threading.Event()
    sandbox = Sandbox(Output("invalid\n"))
    sandbox.terminate_gate = gate
    with pytest.raises(ModalInferenceChannelClientError) as caught:
        _client(sandbox)
    assert type(caught.value.cleanup_lease) is ModalInferenceChannelClient
    assert caught.value.cleanup_lease._sandbox is sandbox
    assert caught.value.cleanup_lease.cleanup_pending is True
    assert caught.value.cleanup_lease is not sandbox
    gate.set()
    assert caught.value.cleanup_lease.close(term_timeout=0, kill_timeout=1) is True
    assert sandbox.terminations == 1


def test_constructor_failure_exposes_exact_completed_cleanup_lease(monkeypatch):
    original_start = ModalInferenceChannelClient._start_termination

    def complete_before_return(client):
        original_start(client)
        assert client._termination_done.wait(1)

    monkeypatch.setattr(
        ModalInferenceChannelClient, "_start_termination", complete_before_return
    )
    sandbox = Sandbox(Output("invalid\n"))
    with pytest.raises(ModalInferenceChannelClientError) as caught:
        _client(sandbox)
    lease = caught.value.cleanup_lease
    assert type(lease) is ModalInferenceChannelClient
    assert lease._sandbox is sandbox
    assert lease.cleanup_pending is False
    assert lease.close(term_timeout=0, kill_timeout=1) is True
    assert sandbox.terminations == 1


@pytest.mark.parametrize(
    "mutation",
    (
        lambda model: model.update(model_kind="adapter"),
        lambda model: model.update(model_revision="A" * 40),
        lambda model: model.update(extra="value"),
    ),
)
def test_malformed_ready_model_is_denied_and_exact_owner_terminated(mutation):
    value = _frame("ready")
    mutation(value["model"])
    raw = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    sandbox = Sandbox(Output(raw))
    with pytest.raises(ModalInferenceChannelClientError) as caught:
        _client(sandbox)
    assert str(caught.value) == "modal_inference_channel_client_invalid"
    assert caught.value.__cause__ is None
    limit = time.monotonic() + 1
    while sandbox.terminations == 0 and time.monotonic() < limit:
        time.sleep(0.001)
    assert sandbox.terminations == 1


def test_expired_absolute_deadline_does_not_start_stdout_read():
    output = Output(_line("ready"))
    sandbox = Sandbox(output)
    with pytest.raises(ModalInferenceChannelClientError):
        _client(sandbox, deadline=time.monotonic())
    assert output.reads == 0
    assert sandbox.terminations == 1


def test_ambiguous_write_timeout_is_not_retried_and_close_only_terminates():
    gate = threading.Event()
    sandbox = Sandbox(Output(_line("ready")))
    sandbox.stdin.block = gate
    client = _client(sandbox, request_timeout_seconds=0.001)
    with pytest.raises(ModalInferenceChannelClientError):
        client.chat(_messages(("user", "first")))
    assert len(sandbox.stdin.writes) == 1
    assert client.close(term_timeout=0.01, kill_timeout=1) is True
    assert len(sandbox.stdin.writes) == 1
    assert sandbox.terminations == 1
    gate.set()


def test_operation_rechecks_deadline_after_receiving_result(monkeypatch):
    sandbox = Sandbox(Output(_line("ready")))
    client = _client(sandbox)
    values = iter((100.0, 102.0))
    monkeypatch.setattr(
        "tuner.execution.providers.modal.inference_channel_client._now",
        lambda: next(values),
    )
    with pytest.raises(TimeoutError):
        client._operation(lambda: "late", 101.0)
    monkeypatch.undo()
    assert client.close(term_timeout=0, kill_timeout=1) is True


def test_stop_control_failure_is_preserved_after_termination():
    failure = KeyboardInterrupt()
    sandbox = Sandbox(Output(_line("ready")))
    client = _client(sandbox)
    sandbox.stdin.drain_error = failure
    with pytest.raises(KeyboardInterrupt) as caught:
        client.close(term_timeout=1, kill_timeout=1)
    assert caught.value is failure
    assert sandbox.terminations == 1


@pytest.mark.parametrize("failure", (KeyboardInterrupt(), SystemExit(9)))
def test_stream_control_failure_preserves_identity_and_starts_cleanup(failure):
    sandbox = Sandbox(Output(_line("ready"), failure))
    client = _client(sandbox)
    with pytest.raises(type(failure)) as caught:
        client.chat(_messages(("user", "first")))
    assert caught.value is failure
    assert client.close(term_timeout=0, kill_timeout=1) is True
    assert sandbox.terminations == 1


def test_module_has_no_modal_sdk_or_adoption_surface():
    import tuner.execution.providers.modal.inference_channel_client as module

    assert "modal" not in module.__dict__
    assert not hasattr(ModalInferenceChannelClient, "from_id")


class _WorkerRuntime:
    def __init__(self):
        self.closed = threading.Event()

    def close(self, *, term_timeout=5.0, kill_timeout=5.0):
        self.closed.set()
        return True

    @property
    def cleanup_pending(self):
        return not self.closed.is_set()


class _WorkerBackend:
    def __init__(self):
        self.calls = []

    def chat(self, messages):
        retained = tuple(dict(item) for item in messages)
        self.calls.append(retained)
        return BackendResponse(
            message=f"answer-{len(self.calls)}",
            raw={"retained": True},
            latency_s=0.001,
        )


class _WorkerInput:
    def __init__(self, values):
        self._values = values

    def readline(self, maximum):
        value = self._values.get()
        assert len(value) <= maximum
        return value


class _WorkerOutput:
    def __init__(self, values):
        self._values = values

    def write(self, value):
        self._values.put(value)
        return len(value)

    def flush(self):
        return None


class _HostInput:
    def __init__(self, values):
        self._values = values

    def write(self, value):
        self._values.put(value)

    def drain(self):
        return None


class _HostOutput:
    def __init__(self, values):
        self._values = values

    def __iter__(self):
        return self

    def __next__(self):
        return self._values.get().decode("utf-8")


class _PipeSandbox:
    def __init__(self, input_values, output_values, worker):
        self._object_id = "sb-pipe"
        self._stdin = _HostInput(input_values)
        self._stdout = _HostOutput(output_values)
        self._worker = worker
        self.terminations = 0

    @property
    def object_id(self):
        return self._object_id

    @property
    def stdin(self):
        return self._stdin

    @property
    def stdout(self):
        return self._stdout

    def terminate(self, *, wait=False):
        assert wait is True
        self.terminations += 1
        self._worker.join(1)
        return 0 if not self._worker.is_alive() else 1


def _pipe_sandbox(*, idle=60.0):
    inputs = Queue()
    outputs = Queue()
    runtime = _WorkerRuntime()
    backend = _WorkerBackend()
    session = ChatSession(
        backend,
        runtime,
        ChatSessionPolicy(1.0, idle, 60.0, 4, 4096),
    )

    def serve():
        serve_modal_chat_channel(
            session,
            _WorkerInput(inputs),
            _WorkerOutput(outputs),
            session_id=SESSION,
            argument_bytes=ARGUMENT,
            max_request_bytes=4096,
            max_response_bytes=4096,
            model=MODEL,
        )

    worker = threading.Thread(target=serve, name="test-modal-chat-worker")
    worker.start()
    return _PipeSandbox(inputs, outputs, worker), backend, runtime, inputs


def test_real_worker_channel_preserves_two_turn_history_and_bounded_cleanup():
    sandbox, backend, runtime, _ = _pipe_sandbox()
    client = _client(sandbox)
    first = client.chat(_messages(("user", "first")))
    second = client.chat(
        _messages(
            ("user", "first"),
            ("assistant", "answer-1"),
            ("user", "second"),
        )
    )
    assert (first.message, second.message) == ("answer-1", "answer-2")
    assert backend.calls == [
        ({"role": "user", "content": "first"},),
        (
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer-1"},
            {"role": "user", "content": "second"},
        ),
    ]
    assert client.close(term_timeout=1, kill_timeout=1) is True
    assert runtime.closed.wait(1)
    assert sandbox.terminations == 1


def test_real_worker_eof_closes_session_and_host_terminates_without_retry():
    sandbox, backend, runtime, inputs = _pipe_sandbox()
    client = _client(sandbox)
    inputs.put(b"")
    limit = time.monotonic() + 1
    while not runtime.closed.is_set() and time.monotonic() < limit:
        time.sleep(0.001)
    assert runtime.closed.is_set()
    assert client.close(term_timeout=1, kill_timeout=1) is True
    assert backend.calls == []
    assert sandbox.terminations == 1
