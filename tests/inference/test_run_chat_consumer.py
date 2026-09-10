from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from Evaluator.chat_session import ChatSession, ChatSessionPolicy
from Evaluator.protocols import BackendResponse
from synaptic_tuner.api.v1.host import APIHost, HostPorts
from synaptic_tuner.api.v1.runs_facade import RunVerification
from tuner.inference.run_chat import (
    PreparedModelIdentity,
    PreparedRunChat,
    open_run_chat,
)
from tuner.inference.retrieved_model import materialize_verified_sft_model
from tuner.inference.serving_target import prepare_serving_target
from tests.inference.test_retrieved_model import _fixture
from tests.inference.test_serving_target import _snapshot


class Clock:
    def now(self) -> str:
        return "2026-09-10T00:00:00Z"


class Training:
    def __getattr__(self, name: str):
        raise AssertionError(f"training operation unexpectedly accessed: {name}")


class Lease:
    def __init__(self) -> None:
        self.close_calls: list[tuple[float, float]] = []
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
        self.calls: list[tuple[dict[str, str], ...]] = []

    def chat(self, messages):
        self.calls.append(tuple(dict(message) for message in messages))
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def _session(backend: Backend, lease: Lease) -> ChatSession:
    return ChatSession(
        backend,
        lease,
        ChatSessionPolicy(1.0, 10.0, 20.0, 4, 4096),
    )


class _Preparer:
    def __init__(self, snapshot) -> None:
        self.snapshot = snapshot

    def prepare(self, *, model_ref: str, revision: str):
        assert (model_ref, revision) == ("example/model", "c" * 40)
        return self.snapshot


class LocalVLLMRuntime:
    """A local adapter which deliberately owns materialization and preparation."""

    def __init__(self, destination: Path, snapshot) -> None:
        self.destination = destination
        self.snapshot = snapshot
        self.backend = Backend()
        self.lease = Lease()
        self.open_calls = []

    @contextmanager
    def open(self, runs, run):
        self.open_calls.append((runs, run))
        retrieved = materialize_verified_sft_model(runs, run, self.destination)
        target = prepare_serving_target(retrieved, _Preparer(self.snapshot))
        assert target.model_path == retrieved.model_path
        with _session(self.backend, self.lease) as session:
            yield PreparedRunChat(
                session=session,
                run=run,
                artifacts=runs.outcome(run).artifacts,
                model=PreparedModelIdentity(
                    retrieved.model_ref,
                    retrieved.model_revision,
                    retrieved.tokenizer_revision,
                    retrieved.model_kind,
                ),
                local_model=retrieved,
            )


class RemoteRuntime:
    """Independent adapter: authenticates metadata but never reads artifact bodies."""

    def __init__(self) -> None:
        self.backend = Backend(BackendResponse("remote", {}, 0.01))
        self.lease = Lease()
        self.starts = 0

    @contextmanager
    def open(self, runs, run):
        verification = runs.reverify(run)
        if not verification.verified or verification.run != run:
            raise ValueError("run reverification failed")
        outcome = runs.outcome(run)
        if outcome.run != run:
            raise ValueError("run outcome mismatch")
        assert any(item.role == "workload_record" for item in outcome.artifacts)
        self.starts += 1
        with _session(self.backend, self.lease) as session:
            yield PreparedRunChat(
                session=session,
                run=run,
                artifacts=outcome.artifacts,
                model=PreparedModelIdentity(
                    "example/model", "c" * 40, "c" * 40, "lora"
                ),
                local_model=None,
            )


def _host(operations) -> APIHost:
    return APIHost(Training(), HostPorts(runs=operations, clock=Clock()))


def test_local_runtime_retains_paths_without_hidden_prompt(tmp_path: Path) -> None:
    run, _, operations = _fixture(tmp_path)
    destination = tmp_path / "retrieved"
    destination.mkdir()
    runtime = LocalVLLMRuntime(destination, _snapshot(tmp_path))

    with open_run_chat(_host(operations).runs, run, runtime=runtime) as prepared:
        assert type(prepared) is PreparedRunChat
        assert prepared.local_model is not None
        assert prepared.model.model_kind == "lora"
        assert prepared.session.chat("hello").message == "answer"

    assert runtime.backend.calls == [({"role": "user", "content": "hello"},)]
    assert runtime.lease.close_calls == [(5.0, 5.0)]
    assert prepared.local_model.model_path.exists()
    assert prepared.local_model.tokenizer_path.exists()


def test_remote_runtime_uses_authenticated_metadata_without_streaming(
    tmp_path: Path,
) -> None:
    run, _, operations = _fixture(tmp_path)
    stream_calls = []
    operations.artifacts = lambda request: stream_calls.append(request)
    runtime = RemoteRuntime()

    with open_run_chat(_host(operations).runs, run, runtime=runtime) as prepared:
        assert prepared.local_model is None
        assert prepared.model == PreparedModelIdentity(
            "example/model", "c" * 40, "c" * 40, "lora"
        )
        assert prepared.session.chat("hello").message == "remote"

    assert runtime.starts == 1
    assert stream_calls == []
    assert runtime.backend.calls == [({"role": "user", "content": "hello"},)]


def test_unauthenticated_run_denied_before_remote_start(tmp_path: Path) -> None:
    run, _, operations = _fixture(tmp_path)
    operations.reverify = lambda value: RunVerification(
        value, False, "2026-09-10T00:00:00Z"
    )
    runtime = RemoteRuntime()

    with pytest.raises(ValueError, match="reverification failed"):
        with open_run_chat(_host(operations).runs, run, runtime=runtime):
            pytest.fail("unauthenticated run entered chat context")

    assert runtime.starts == 0
    assert runtime.backend.calls == []
    assert runtime.lease.close_calls == []
