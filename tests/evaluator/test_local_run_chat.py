from __future__ import annotations

from contextlib import contextmanager
from dataclasses import FrozenInstanceError

import pytest

from Evaluator import local_run_chat, vllm_runtime
from Evaluator.chat_session import ChatSession, ChatSessionPolicy
from Evaluator.local_run_chat import LocalVLLMRunChatRuntime
from Evaluator.protocols import BackendResponse
from synaptic_tuner.api.v1.runs_facade import RunsAPI
from tests.evaluator.test_verified_vllm_chat import _real_target
from tests.inference.test_run_chat import _case


def _adapter(tmp_path, **options):
    return LocalVLLMRunChatRuntime(
        ChatSessionPolicy(1, 10, 20, 2, 4096),
        tmp_path,
        {},
        destination=options.pop("destination", tmp_path / "output"),
        **options,
    )


@pytest.mark.parametrize("kind", ["full", "lora"])
def test_adapter_preserves_target_and_delegates_without_extra_validation(
    tmp_path, monkeypatch, kind
):
    target = _real_target(tmp_path, kind)
    calls = []
    monkeypatch.setattr(
        type(target), "validate", lambda self: pytest.fail("extra validation")
    )
    runs, run = RunsAPI(object()), target.retrieved.run
    stages = []

    def materialize(actual_runs, actual_run, destination):
        assert actual_runs is runs and actual_run is run
        assert destination == tmp_path / "output"
        stages.append("materialize")
        return target.retrieved

    def prepare(retrieved, preparer):
        assert retrieved is target.retrieved and preparer is None
        stages.append("prepare")
        return target

    monkeypatch.setattr(local_run_chat, "materialize_verified_sft_model", materialize)
    monkeypatch.setattr(local_run_chat, "prepare_serving_target", prepare)

    class Client:
        def chat(self, messages):
            return BackendResponse("unused", {}, 0.0)

    class Lease:
        cleanup_pending = True

        def close(self, **kwargs):
            self.cleanup_pending = False
            return True

    @contextmanager
    def fake_open(startup, policy, **kwargs):
        calls.append((startup, policy, kwargs))
        stages.append("open")
        with ChatSession(Client(), Lease(), policy) as session:
            yield session

    monkeypatch.setattr(local_run_chat, "verified_vllm_chat", fake_open)
    adapter = _adapter(
        tmp_path,
        startup_options={
            "port": 9123,
            "tensor_parallel_size": 2,
            "python_executable": "/opt/inference/bin/python3",
        },
        max_request_bytes=8192,
    )
    with adapter.open(runs, run) as prepared:
        assert type(prepared.session) is ChatSession
        assert prepared.run == run
        assert prepared.local_model is target.retrieved
        assert prepared.artifacts == target.retrieved.artifacts
        assert prepared.model.model_kind == kind
    assert stages == ["materialize", "prepare", "open"]
    startup, policy, kwargs = calls[0]
    assert len(calls) == 1
    assert type(startup.source) is vllm_runtime.VerifiedLocalVLLMSource
    assert startup.source.target is target
    assert startup.port == 9123 and startup.tensor_parallel_size == 2
    assert startup.python_executable == "/opt/inference/bin/python3"
    assert startup.served_model_name == "trained"
    assert policy is adapter.policy
    assert kwargs["environment"]["HF_HUB_OFFLINE"] == "1"
    assert kwargs["max_tokens"] == 128
    assert kwargs["max_request_bytes"] == 8192


@pytest.mark.parametrize("bound", (None, True, 0, -1, 1.5, 64 * 1024 * 1024 + 1))
def test_adapter_rejects_invalid_request_limit_before_open(tmp_path, bound):
    with pytest.raises(ValueError):
        _adapter(tmp_path, max_request_bytes=bound)


def test_adapter_snapshots_mutable_config_and_redacts_environment(tmp_path):
    env = {"PATH": "/fixture/path"}
    options = {"port": 9123}
    adapter = LocalVLLMRunChatRuntime(
        ChatSessionPolicy(1, 10, 20, 2, 4096),
        tmp_path,
        env,
        destination=tmp_path / "output",
        startup_options=options,
    )
    env["PATH"] = "/changed"
    options["port"] = 9999
    assert adapter.environment["PATH"] == "/fixture/path"
    assert adapter.startup_options["port"] == 9123
    assert "/fixture/path" not in repr(adapter)
    with pytest.raises(TypeError):
        adapter.startup_options["port"] = 9999
    with pytest.raises(FrozenInstanceError):
        adapter.served_model_name = "changed"


@pytest.mark.parametrize("name", ["HF_TOKEN", "HTTPS_PROXY", "PYTHONPATH"])
def test_adapter_rejects_forbidden_child_environment_before_open(tmp_path, name):
    with pytest.raises(ValueError):
        LocalVLLMRunChatRuntime(
            ChatSessionPolicy(1, 10, 20, 2, 4096),
            tmp_path,
            {name: "fixture"},
            destination=tmp_path / "output",
        )


@pytest.mark.parametrize(
    "options",
    [{"source": None}, {"served_model_name": "other"}, {"port": []}, {"unknown": 1}],
)
def test_adapter_rejects_overrides_or_nonprimitive_options(tmp_path, options):
    with pytest.raises((TypeError, ValueError)):
        _adapter(tmp_path, startup_options=options)


def test_adapter_requires_exact_run_inputs(tmp_path):
    with pytest.raises(TypeError):
        with _adapter(tmp_path).open(object(), object()):
            pytest.fail("opened")


def test_adapter_rejects_invalid_preparer_at_construction(tmp_path):
    with pytest.raises(TypeError):
        _adapter(tmp_path, preparer=object())


def test_local_reverification_denial_never_prepares_or_starts(tmp_path, monkeypatch):
    from synaptic_tuner.api.v1.runs_facade import RunVerification

    runs, run, destination, _, operations = _case(tmp_path, "full")
    operations.reverify = lambda requested: RunVerification(
        requested, False, "2026-09-10T00:00:00Z"
    )
    monkeypatch.setattr(
        local_run_chat,
        "prepare_serving_target",
        lambda *a, **k: pytest.fail("prepared rejected run"),
    )
    monkeypatch.setattr(
        local_run_chat,
        "verified_vllm_chat",
        lambda *a, **k: pytest.fail("started rejected run"),
    )
    with pytest.raises(ValueError, match="reverification"):
        with _adapter(tmp_path, destination=destination).open(runs, run):
            pytest.fail("opened")
    assert list(destination.iterdir()) == []


def test_missing_local_destination_fails_before_run_reads(tmp_path):
    runs, run, _, _, operations = _case(tmp_path, "full")
    operations.reverify = lambda value: pytest.fail("read before private-root check")
    with pytest.raises((OSError, ValueError)):
        with _adapter(tmp_path).open(runs, run):
            pytest.fail("opened")


@pytest.mark.parametrize(
    "options",
    [
        {"startup_options": {"port": 0}},
        {"max_tokens": 0},
        {"temperature": float("nan")},
    ],
)
def test_existing_validation_rejects_invalid_ranges_before_spawn(
    tmp_path, monkeypatch, options
):
    runs, run, destination, _, _ = _case(tmp_path, "full")
    monkeypatch.setattr(vllm_runtime, "_spawn", lambda *a, **k: pytest.fail("spawned"))
    adapter = _adapter(tmp_path, destination=destination, **options)
    with pytest.raises((TypeError, ValueError)):
        with adapter.open(runs, run):
            pytest.fail("yielded")
