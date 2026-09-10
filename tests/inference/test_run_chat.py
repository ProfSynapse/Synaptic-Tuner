from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from Evaluator.chat_session import ChatSession, ChatSessionPolicy
from Evaluator.protocols import BackendResponse
from synaptic_tuner.api.v1.runs_facade import RunsAPI
from tests.inference.test_retrieved_model import _fixture, _safe, _tar
from tests.inference.test_serving_target import _snapshot
from tuner.inference.run_chat import PreparedRunChat, open_run_chat


class _Client:
    def chat(self, messages):
        return BackendResponse(message="ok", raw={}, latency_s=0.0)


class _Lease:
    cleanup_pending = True

    def close(self, **kwargs):
        self.cleanup_pending = False
        return True


def _session() -> ChatSession:
    return ChatSession(
        _Client(),
        _Lease(),
        ChatSessionPolicy(1, 60, 60, 4, 4096),
    )


_DEFAULT_SESSION = object()


class _Runtime:
    def __init__(
        self,
        session: object = _DEFAULT_SESSION,
        failure: BaseException | None = None,
    ):
        self.session = session
        self.failure = failure
        self.targets = []
        self.exits = 0
        self.sessions = 0

    @contextmanager
    def open(self, target):
        self.targets.append(target)
        session = self.session
        if session is _DEFAULT_SESSION:
            session = _session()
            self.session = session
            self.sessions += 1
        try:
            if self.failure is not None:
                raise self.failure
            yield session
        finally:
            self.exits += 1
            if type(session) is ChatSession:
                session.close()
                session.wait_closed()


def _case(tmp_path: Path, kind: str):
    destination = tmp_path / "retrieved"
    destination.mkdir()
    run, values, operations = _fixture(destination)
    if kind == "full":
        values["final_model"] = _tar(
            (
                ("config.json", b'{"model_type":"fixture"}'),
                ("model.safetensors", _safe()),
            )
        )
        operations.inventory = tuple(
            replace(
                item,
                size_bytes=len(values[item.role]),
                sha256=hashlib.sha256(values[item.role]).hexdigest(),
            )
            for item in operations.inventory
        )
        preparer = None
    else:
        snapshot = _snapshot(tmp_path)

        class Preparer:
            def __init__(self):
                self.calls = []

            def prepare(self, *, model_ref, revision):
                self.calls.append((model_ref, revision))
                return snapshot

        preparer = Preparer()
    return RunsAPI(operations), run, destination, preparer, operations


@pytest.mark.parametrize("kind", ("full", "lora"))
def test_real_run_materialization_preparation_and_one_runtime_open(tmp_path, kind):
    runs, run, destination, preparer, _ = _case(tmp_path, kind)
    runtime = _Runtime()

    with open_run_chat(
        runs,
        run,
        destination=destination,
        runtime=runtime,
        preparer=preparer,
    ) as opened:
        assert type(opened) is PreparedRunChat
        assert opened.session is runtime.session
        assert opened.target is runtime.targets[0]
        assert opened.retrieved.model_kind == kind
        assert opened.target.retrieved is opened.retrieved
        assert opened.session.chat("hello").message == "ok"
        attempt = opened.retrieved.root / opened.retrieved.attempt
        assert attempt.is_dir()
        if kind == "lora":
            assert preparer.calls == [("example/model", "c" * 40)]
            assert opened.target.base_snapshot is not None
        else:
            assert opened.target.base_snapshot is None

    assert len(runtime.targets) == 1
    assert runtime.exits == 1
    assert attempt.is_dir()


def test_invalid_configuration_fails_before_run_reads(tmp_path):
    runs, run, destination, _, operations = _case(tmp_path, "lora")
    calls = []
    operations.reverify = lambda value: calls.append(value)

    class MissingOpen:
        pass

    with pytest.raises(TypeError, match="callable open"):
        with open_run_chat(runs, run, destination=destination, runtime=MissingOpen()):
            pytest.fail("yielded")
    assert calls == []

    unopened = _Runtime()
    with pytest.raises(TypeError, match="PinnedModelPreparer"):
        with open_run_chat(
            runs,
            run,
            destination=destination,
            runtime=unopened,
            preparer=object(),
        ):
            pytest.fail("yielded")
    assert calls == []
    assert unopened.targets == [] and unopened.sessions == 0


def test_instance_open_shadow_fails_before_run_reads(tmp_path):
    runs, run, destination, _, operations = _case(tmp_path, "lora")
    calls = []
    operations.reverify = lambda value: calls.append(value)
    runtime = _Runtime()
    runtime.open = object()
    with pytest.raises(TypeError, match="callable open"):
        with open_run_chat(runs, run, destination=destination, runtime=runtime):
            pytest.fail("yielded")
    assert calls == []
    assert runtime.targets == [] and runtime.sessions == 0


def test_wrong_session_type_is_rejected_inside_runtime_context(tmp_path):
    runs, run, destination, preparer, _ = _case(tmp_path, "lora")
    runtime = _Runtime(session=object())
    with pytest.raises(TypeError, match="exact ChatSession"):
        with open_run_chat(
            runs,
            run,
            destination=destination,
            runtime=runtime,
            preparer=preparer,
        ):
            pytest.fail("yielded")
    assert len(runtime.targets) == 1
    assert runtime.exits == 1


@pytest.mark.parametrize("failure", (RuntimeError("runtime"), KeyboardInterrupt()))
def test_runtime_failure_is_preserved_without_retry(tmp_path, failure):
    runs, run, destination, preparer, _ = _case(tmp_path, "lora")
    runtime = _Runtime(failure=failure)
    with pytest.raises(type(failure)) as caught:
        with open_run_chat(
            runs,
            run,
            destination=destination,
            runtime=runtime,
            preparer=preparer,
        ):
            pytest.fail("yielded")
    assert caught.value is failure
    assert len(runtime.targets) == 1
    assert runtime.exits == 1


def test_reverification_denial_prevents_preparation_and_runtime(tmp_path):
    runs, run, destination, preparer, operations = _case(tmp_path, "lora")
    operations.reverify = lambda value: replace(
        type(operations).reverify(operations, value), verified=False
    )
    runtime = _Runtime()
    with pytest.raises(ValueError, match="reverification"):
        with open_run_chat(
            runs,
            run,
            destination=destination,
            runtime=runtime,
            preparer=preparer,
        ):
            pytest.fail("yielded")
    assert preparer.calls == []
    assert runtime.targets == []
    assert runtime.sessions == 0
