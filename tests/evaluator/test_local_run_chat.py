from __future__ import annotations

from contextlib import contextmanager
from dataclasses import FrozenInstanceError

import pytest

from Evaluator import local_run_chat, vllm_runtime
from Evaluator.chat_session import ChatSessionPolicy
from Evaluator.local_run_chat import LocalVLLMRunChatRuntime
from tests.evaluator.test_verified_vllm_chat import _real_target


def _adapter(tmp_path, **options):
    return LocalVLLMRunChatRuntime(
        ChatSessionPolicy(1, 10, 20, 2, 4096), tmp_path, {}, **options
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

    @contextmanager
    def fake_open(startup, policy, **kwargs):
        calls.append((startup, policy, kwargs))
        yield "session"

    monkeypatch.setattr(local_run_chat, "verified_vllm_chat", fake_open)
    adapter = _adapter(
        tmp_path, startup_options={"port": 9123, "tensor_parallel_size": 2}
    )
    with adapter.open(target) as session:
        assert session == "session"
    startup, policy, kwargs = calls[0]
    assert len(calls) == 1
    assert type(startup.source) is vllm_runtime.VerifiedLocalVLLMSource
    assert startup.source.target is target
    assert startup.port == 9123 and startup.tensor_parallel_size == 2
    assert startup.served_model_name == "trained"
    assert policy is adapter.policy
    assert kwargs["environment"]["HF_HUB_OFFLINE"] == "1"
    assert kwargs["max_tokens"] == 128


def test_adapter_snapshots_mutable_config_and_redacts_environment(tmp_path):
    env = {"PATH": "/fixture/path"}
    options = {"port": 9123}
    adapter = LocalVLLMRunChatRuntime(
        ChatSessionPolicy(1, 10, 20, 2, 4096),
        tmp_path,
        env,
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
            ChatSessionPolicy(1, 10, 20, 2, 4096), tmp_path, {name: "fixture"}
        )


@pytest.mark.parametrize(
    "options",
    [{"source": None}, {"served_model_name": "other"}, {"port": []}, {"unknown": 1}],
)
def test_adapter_rejects_overrides_or_nonprimitive_options(tmp_path, options):
    with pytest.raises((TypeError, ValueError)):
        _adapter(tmp_path, startup_options=options)


def test_adapter_requires_exact_target(tmp_path):
    with pytest.raises(TypeError):
        _adapter(tmp_path).open(object())


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
    target = _real_target(tmp_path, "full")
    monkeypatch.setattr(vllm_runtime, "_spawn", lambda *a, **k: pytest.fail("spawned"))
    adapter = _adapter(tmp_path, **options)
    with pytest.raises((TypeError, ValueError)):
        with adapter.open(target):
            pytest.fail("yielded")
