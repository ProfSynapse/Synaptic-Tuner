from __future__ import annotations
from contextlib import contextmanager
from dataclasses import replace
import hashlib
from pathlib import Path
import pytest
from Evaluator.chat_session import ChatSession, ChatSessionPolicy
from Evaluator.protocols import BackendResponse
from synaptic_tuner.api.v1.results import TrainingRunRef, VerifiedArtifact
from synaptic_tuner.api.v1.runs_facade import RunsAPI
from tests.inference.test_retrieved_model import _fixture, _safe, _tar
from tests.inference.test_serving_target import _snapshot
from tuner.inference.retrieved_model import ROLES, materialize_verified_sft_model
from tuner.inference.run_chat import (
    PreparedModelIdentity,
    PreparedRunChat,
    open_run_chat,
)


class _Client:
    def chat(self, messages):
        return BackendResponse(message="ok", raw={}, latency_s=0.0)


class _Lease:
    cleanup_pending = True

    def close(self, **kwargs):
        self.cleanup_pending = False
        return True


def _session():
    return ChatSession(_Client(), _Lease(), ChatSessionPolicy(1, 60, 60, 4, 4096))


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


def _artifacts():
    return tuple(VerifiedArtifact(role, "a" * 64, 1) for role in ROLES)


def _model(kind="lora"):
    return PreparedModelIdentity("example/model", "c" * 40, "c" * 40, kind)


class _Runtime:
    def __init__(self, prepared=None, failure=None):
        self.prepared, self.failure, self.calls, self.exits = prepared, failure, [], 0

    @contextmanager
    def open(self, runs, run):
        self.calls.append((runs, run))
        try:
            if self.failure is not None:
                raise self.failure
            yield self.prepared
        finally:
            self.exits += 1
            if type(self.prepared) is PreparedRunChat:
                self.prepared.session.close()
                self.prepared.session.wait_closed()


def test_remote_runtime_delegates_once_without_runs_or_local_filesystem(tmp_path):
    class NoCalls:
        def __getattr__(self, name):
            raise AssertionError(name)

    runs, run = RunsAPI(NoCalls()), TrainingRunRef("run-1", "project-1")
    prepared = PreparedRunChat(_session(), run, _artifacts(), _model())
    runtime = _Runtime(prepared)
    with open_run_chat(runs, run, runtime=runtime) as opened:
        assert opened is not prepared and opened == prepared
        assert opened.local_model is None
        assert not list(tmp_path.iterdir())
    assert runtime.calls[0][0] is runs and runtime.calls[0][1] == run
    assert runtime.calls[0][1] is not run and runtime.exits == 1


@pytest.mark.parametrize("runtime", (object(),))
def test_invalid_runtime_fails_before_delegation(runtime):
    with pytest.raises(TypeError, match="callable open"):
        with open_run_chat(
            RunsAPI(object()), TrainingRunRef("r", "p"), runtime=runtime
        ):
            pytest.fail("yielded")


def test_instance_open_property_is_not_invoked():
    class Runtime:
        @property
        def open(self):
            raise AssertionError("invoked")

    with pytest.raises(TypeError, match="callable open"):
        with open_run_chat(
            RunsAPI(object()), TrainingRunRef("r", "p"), runtime=Runtime()
        ):
            pytest.fail("yielded")


@pytest.mark.parametrize("value", (object(), _session()))
def test_wrong_result_type_tears_down_context(value):
    runtime = _Runtime(value)
    with pytest.raises(TypeError, match="exact PreparedRunChat"):
        with open_run_chat(
            RunsAPI(object()), TrainingRunRef("r", "p"), runtime=runtime
        ):
            pytest.fail("yielded")
    assert runtime.exits == 1
    if type(value) is ChatSession:
        value.close()
        value.wait_closed()


def test_wrong_run_tears_down_context():
    prepared = PreparedRunChat(
        _session(), TrainingRunRef("other", "p"), _artifacts(), _model()
    )
    runtime = _Runtime(prepared)
    with pytest.raises(ValueError, match="requested run"):
        with open_run_chat(
            RunsAPI(object()), TrainingRunRef("r", "p"), runtime=runtime
        ):
            pytest.fail("yielded")
    assert runtime.exits == 1


def test_runtime_cannot_mutate_presented_run():
    requested = TrainingRunRef("r", "p")
    prepared = PreparedRunChat(_session(), requested, _artifacts(), _model())

    class Runtime(_Runtime):
        @contextmanager
        def open(self, runs, run):
            self.calls.append((runs, run))
            object.__setattr__(run, "run_id", "changed")
            try:
                yield prepared
            finally:
                self.exits += 1
                prepared.session.close()
                prepared.session.wait_closed()

    runtime = Runtime(prepared)
    with pytest.raises(ValueError, match="run changed"):
        with open_run_chat(RunsAPI(object()), requested, runtime=runtime):
            pytest.fail("yielded")
    assert requested == TrainingRunRef("r", "p")
    assert runtime.exits == 1


@pytest.mark.parametrize("failure", (RuntimeError("runtime"), KeyboardInterrupt()))
def test_runtime_failure_is_preserved_without_retry(failure):
    runtime = _Runtime(failure=failure)
    with pytest.raises(type(failure)) as caught:
        with open_run_chat(
            RunsAPI(object()), TrainingRunRef("r", "p"), runtime=runtime
        ):
            pytest.fail("yielded")
    assert caught.value is failure
    assert len(runtime.calls) == 1 and runtime.exits == 1


def test_local_projection_mismatch_is_cheaply_rejected(tmp_path):
    runs, run, destination, _, _ = _case(tmp_path, "lora")
    local = materialize_verified_sft_model(runs, run, destination)
    session = _session()
    try:
        with pytest.raises(ValueError, match="local model projection differs"):
            PreparedRunChat(
                session,
                run,
                local.artifacts,
                PreparedModelIdentity(
                    local.model_ref,
                    "d" * 40,
                    local.tokenizer_revision,
                    local.model_kind,
                ),
                local,
            )
    finally:
        session.close()
        session.wait_closed()


@pytest.mark.parametrize(
    "artifacts",
    (
        _artifacts()[:-1],
        tuple(reversed(_artifacts())),
        tuple(
            replace(x, role="other") if x.role == ROLES[0] else x for x in _artifacts()
        ),
    ),
)
def test_inventory_must_be_exact_canonical_five(artifacts):
    session = _session()
    try:
        with pytest.raises(ValueError, match="canonical SFT inventory"):
            PreparedRunChat(session, TrainingRunRef("r", "p"), artifacts, _model())
    finally:
        session.close()
        session.wait_closed()


def test_model_kind_requires_exact_string():
    class EqualString(str):
        pass

    with pytest.raises(ValueError, match="model_kind"):
        PreparedModelIdentity("example/model", "c" * 40, "c" * 40, EqualString("lora"))


def test_artifacts_require_positive_sizes():
    artifacts = tuple(
        replace(item, size_bytes=0) if item.role == ROLES[0] else item
        for item in _artifacts()
    )
    session = _session()
    try:
        with pytest.raises(ValueError, match="canonical SFT inventory"):
            PreparedRunChat(session, TrainingRunRef("r", "p"), artifacts, _model())
    finally:
        session.close()
        session.wait_closed()
