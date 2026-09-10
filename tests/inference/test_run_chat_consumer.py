from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path

import pytest

from Evaluator.chat_session import ChatSession, ChatSessionError, ChatSessionPolicy
from Evaluator.protocols import BackendResponse
from synaptic_tuner.api.v1.host import APIHost, HostPorts
from synaptic_tuner.api.v1.runs_facade import RunVerification
from tuner.inference.run_chat import PreparedRunChat, open_run_chat
from tests.inference.test_retrieved_model import _fixture, _safe, _tar
from tests.inference.test_serving_target import _snapshot


class Clock:
    def now(self) -> str:
        return "2026-09-10T00:00:00Z"


class Training:
    def __getattr__(self, name: str):
        raise AssertionError(f"training operation unexpectedly accessed: {name}")


class Lease:
    def __init__(self) -> None:
        self.close_calls = []
        self._pending = True

    @property
    def cleanup_pending(self) -> bool:
        return self._pending

    def close(self, *, term_timeout: float = 5.0, kill_timeout: float = 5.0) -> bool:
        self.close_calls.append((term_timeout, kill_timeout))
        self._pending = False
        return True


class Backend:
    def __init__(self, response: object = None) -> None:
        self.response = response or BackendResponse("answer", {}, 0.01)
        self.calls = []

    def chat(self, messages):
        self.calls.append(tuple(dict(message) for message in messages))
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class LocalVLLMRuntime:
    """One consumer-owned adapter; it has no provider selector or registry."""

    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self.lease = Lease()
        self.targets = []

    @contextmanager
    def open(self, target):
        self.targets.append(target)
        with ChatSession(
            self.backend,
            self.lease,
            ChatSessionPolicy(1.0, 10.0, 20.0, 4, 4096),
        ) as session:
            yield session


class IndependentLocalRuntime:
    """A separately implemented local adapter accepted through the same seam."""

    def __init__(self) -> None:
        self.backend = Backend(BackendResponse("independent", {}, 0.01))
        self.lease = Lease()
        self.opened_paths = []

    @contextmanager
    def open(self, target):
        self.opened_paths.append(
            (target.model_path, target.tokenizer_path, target.base_model_path)
        )
        session = ChatSession(
            self.backend,
            self.lease,
            ChatSessionPolicy(1.0, 10.0, 20.0, 4, 4096),
        )
        try:
            yield session
        finally:
            session.close()
            session.wait_closed(1.0)


class Preparer:
    def __init__(self, snapshot) -> None:
        self.snapshot = snapshot
        self.calls = []

    def prepare(self, *, model_ref: str, revision: str):
        self.calls.append((model_ref, revision))
        return self.snapshot


def _host(operations) -> APIHost:
    return APIHost(Training(), HostPorts(runs=operations, clock=Clock()))


def _destination(tmp_path: Path, name: str) -> Path:
    destination = tmp_path / name
    destination.mkdir()
    return destination


def _make_full(operations) -> None:
    payload = _tar(
        (("config.json", b'{"model_type":"fixture"}'), ("model.safetensors", _safe()))
    )
    operations.values["final_model"] = payload
    operations.inventory = tuple(
        (
            type(artifact)(
                artifact.role,
                hashlib.sha256(payload).hexdigest(),
                len(payload),
            )
            if artifact.role == "final_model"
            else artifact
        )
        for artifact in operations.inventory
    )


def test_host_runs_opens_lora_chat_once_without_hidden_prompt(tmp_path: Path) -> None:
    run, _, operations = _fixture(tmp_path)
    host = _host(operations)
    snapshot = _snapshot(tmp_path)
    preparer = Preparer(snapshot)
    backend = Backend()
    runtime = LocalVLLMRuntime(backend)

    with open_run_chat(
        host.runs,
        run,
        destination=_destination(tmp_path, "retrieved"),
        runtime=runtime,
        preparer=preparer,
    ) as prepared:
        assert type(prepared) is PreparedRunChat
        assert prepared.retrieved.model_kind == "lora"
        assert prepared.target.base_snapshot is snapshot
        assert prepared.target.model_path == prepared.retrieved.model_path
        assert prepared.target.tokenizer_path == prepared.retrieved.tokenizer_path
        assert prepared.session.chat("hello").message == "answer"

    assert preparer.calls == [("example/model", "c" * 40)]
    assert runtime.targets == [prepared.target]
    assert backend.calls == [({"role": "user", "content": "hello"},)]
    assert runtime.lease.close_calls == [(5.0, 5.0)]
    assert prepared.retrieved.model_path.exists()
    assert prepared.retrieved.tokenizer_path.exists()


def test_full_model_uses_no_preparer_and_independent_runtime(tmp_path: Path) -> None:
    run, _, operations = _fixture(tmp_path)
    _make_full(operations)
    runtime = IndependentLocalRuntime()

    with open_run_chat(
        _host(operations).runs,
        run,
        destination=_destination(tmp_path, "full"),
        runtime=runtime,
    ) as prepared:
        assert prepared.retrieved.model_kind == "full"
        assert prepared.target.base_snapshot is None
        assert prepared.session.chat("plain input").message == "independent"

    assert len(runtime.opened_paths) == 1
    model, tokenizer, base = runtime.opened_paths[0]
    assert model == prepared.retrieved.model_path
    assert tokenizer == prepared.retrieved.tokenizer_path
    assert base is None
    assert runtime.backend.calls == [({"role": "user", "content": "plain input"},)]
    assert runtime.lease.close_calls == [(5.0, 5.0)]


def test_backend_error_closes_owned_runtime_and_retains_files(tmp_path: Path) -> None:
    run, _, operations = _fixture(tmp_path)
    runtime = LocalVLLMRuntime(Backend(RuntimeError("private backend detail")))
    preparer = Preparer(_snapshot(tmp_path))

    with pytest.raises(ChatSessionError, match="backend chat request failed"):
        with open_run_chat(
            _host(operations).runs,
            run,
            destination=_destination(tmp_path, "error"),
            runtime=runtime,
            preparer=preparer,
        ) as prepared:
            prepared.session.chat("hello")

    assert runtime.lease.close_calls == [(5.0, 5.0)]
    assert prepared.retrieved.model_path.exists()
    assert prepared.retrieved.tokenizer_path.exists()


def test_unauthenticated_run_denied_before_preparer_or_runtime(tmp_path: Path) -> None:
    run, _, operations = _fixture(tmp_path)
    operations.reverify = lambda value: RunVerification(
        value, False, "2026-09-10T00:00:00Z"
    )
    preparer = Preparer(_snapshot(tmp_path))
    runtime = LocalVLLMRuntime(Backend())

    with pytest.raises(ValueError, match="reverification failed"):
        with open_run_chat(
            _host(operations).runs,
            run,
            destination=_destination(tmp_path, "denied"),
            runtime=runtime,
            preparer=preparer,
        ):
            pytest.fail("unauthenticated run entered chat context")

    assert preparer.calls == []
    assert runtime.targets == []
    assert runtime.backend.calls == []
    assert runtime.lease.close_calls == []
