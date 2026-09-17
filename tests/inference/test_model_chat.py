"""Model-first chat needs neither a training API nor a retained training run."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from Evaluator.chat_session import ChatSessionPolicy
from Evaluator.protocols import BackendResponse
from Evaluator.vllm_runtime import (
    ExplicitNetworkLoRA,
    ExplicitNetworkVLLMSource,
    VLLMRuntimeLease,
    VLLMStartupSpec,
)
from tuner.inference import model_chat


@pytest.fixture
def case(monkeypatch):
    class Process:
        cleanup_pending = True
        calls = 0
        resolved = True

        def close(self, **kwargs):
            self.calls += 1
            self.cleanup_pending = not self.resolved
            return self.resolved

    process = Process()
    runtime = VLLMRuntimeLease(
        process, host="127.0.0.1", port=8000, served_model_name="selected-model"
    )
    captured = {"starts": [], "requests": []}

    def start(spec, **kwargs):
        captured["starts"].append((spec, kwargs))
        return runtime

    class Client:
        def __init__(self, settings, **kwargs):
            captured["client"] = kwargs

        def chat(self, messages):
            captured["requests"].append(messages)
            return BackendResponse("hello", {}, 0.01)

    monkeypatch.setattr(model_chat, "start_vllm_runtime", start)
    monkeypatch.setattr(model_chat, "VLLMClient", Client)
    startup = VLLMStartupSpec(
        ExplicitNetworkVLLMSource("owner/model", "a" * 40), "selected-model"
    )
    return SimpleNamespace(
        startup=startup,
        policy=ChatSessionPolicy(1, 10, 20, 2, 4096),
        process=process,
        runtime=runtime,
        captured=captured,
    )


def test_direct_model_chat_no_training_state_no_hidden_prompt(
    case, tmp_path, monkeypatch
):
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "fixture-should-not-be-inherited")
    with model_chat.open_model_chat(
        case.startup, case.policy, cwd=tmp_path, environment={}
    ) as chat:
        assert case.captured["requests"] == []
        assert chat.chat("hello").message == "hello"
    assert len(case.captured["starts"]) == 1
    assert case.captured["starts"][0][1]["environment"] == {
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"
    }
    assert len(case.captured["requests"]) == 1
    assert case.captured["client"]["retries"] == 0
    assert case.captured["client"]["trust_environment"] is False
    assert case.captured["client"]["allow_redirects"] is False
    assert case.process.calls == 1
    assert not case.runtime.cleanup_pending


@pytest.mark.parametrize("revision", [None, "main", "", "a" * 39])
def test_floating_hub_selection_rejected_before_start(case, tmp_path, revision):
    startup = replace(
        case.startup, source=ExplicitNetworkVLLMSource("owner/model", revision)
    )
    with pytest.raises(ValueError):
        with model_chat.open_model_chat(
            startup, case.policy, cwd=tmp_path, environment={}
        ):
            pytest.fail("unselected model started")
    assert case.captured["starts"] == []


def test_existing_model_directory_is_independent_of_training_provenance(case, tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    startup = replace(case.startup, source=ExplicitNetworkVLLMSource(str(model)))
    with model_chat.open_model_chat(startup, case.policy, cwd=tmp_path, environment={}):
        pass
    assert case.captured["starts"][0][0].source.model_ref == str(model)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tokens": True},
        {"max_tokens": 0},
        {"max_request_bytes": 0},
        {"temperature": float("nan")},
        {"top_p": 0},
        {"deadline": float("inf")},
    ],
)
def test_invalid_bounds_precede_start(case, tmp_path, kwargs):
    with pytest.raises((TypeError, ValueError)):
        with model_chat.open_model_chat(
            case.startup, case.policy, cwd=tmp_path, environment={}, **kwargs
        ):
            pytest.fail("invalid bounds started")
    assert case.captured["starts"] == []


def test_client_failure_closes_runtime_and_hides_exception(case, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("private-backend-detail")

    monkeypatch.setattr(model_chat, "VLLMClient", fail)
    with pytest.raises(model_chat.ModelChatError) as caught:
        with model_chat.open_model_chat(
            case.startup, case.policy, cwd=tmp_path, environment={}
        ):
            pytest.fail("client failure yielded")
    assert "private-backend-detail" not in str(caught.value)
    assert case.process.calls == 1


def test_unresolved_cleanup_retains_exact_lease(case, tmp_path):
    case.process.resolved = False
    with pytest.raises(model_chat.ModelChatError) as caught:
        with model_chat.open_model_chat(
            case.startup, case.policy, cwd=tmp_path, environment={}
        ):
            raise ValueError("private")
    assert caught.value.cleanup_lease is case.runtime
    assert case.runtime.cleanup_pending


def test_expiry_during_startup_does_not_yield(case, tmp_path, monkeypatch):
    now = [100.0]
    start = model_chat.start_vllm_runtime

    def slow(*args, **kwargs):
        now[0] = 102.0
        return start(*args, **kwargs)

    monkeypatch.setattr(model_chat.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(model_chat, "start_vllm_runtime", slow)
    with pytest.raises(model_chat.ModelChatError):
        with model_chat.open_model_chat(
            case.startup, case.policy, cwd=tmp_path, environment={}, deadline=101.0
        ):
            pytest.fail("expired startup yielded")
    assert case.captured["requests"] == []
    assert case.process.calls == 1


@pytest.mark.parametrize("invalid", ["symlink", "relative", "type", "name"])
def test_adapter_is_canonical_and_bound_before_start(case, tmp_path, invalid):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    link = tmp_path / "link"
    link.symlink_to(adapter, target_is_directory=True)
    selected = {
        "symlink": link,
        "relative": adapter.relative_to(tmp_path),
        "type": str(adapter),
        "name": adapter,
    }[invalid]
    lora = ExplicitNetworkLoRA(
        "other" if invalid == "name" else "selected-model", selected
    )
    startup = replace(case.startup, source=replace(case.startup.source, lora=lora))
    with pytest.raises(ValueError):
        with model_chat.open_model_chat(
            startup, case.policy, cwd=tmp_path, environment={}
        ):
            pytest.fail("invalid adapter started")
    assert case.captured["starts"] == []


def test_immutable_interrupt_is_preserved_when_cleanup_unresolved(case, tmp_path):
    class ImmutableInterrupt(KeyboardInterrupt):
        def __setattr__(self, name, value):
            if name == "cleanup_lease":
                raise AttributeError("read only")
            super().__setattr__(name, value)

    primary = ImmutableInterrupt()
    case.process.resolved = False
    with pytest.raises(ImmutableInterrupt) as caught:
        with model_chat.open_model_chat(
            case.startup, case.policy, cwd=tmp_path, environment={}
        ):
            raise primary
    assert caught.value is primary
    assert case.runtime.cleanup_pending


def test_startup_failure_preserves_lower_level_cleanup_lease(
    case, tmp_path, monkeypatch
):
    owner = object()

    def fail(*args, **kwargs):
        error = RuntimeError("private")
        error.cleanup_lease = owner
        raise error

    monkeypatch.setattr(model_chat, "start_vllm_runtime", fail)
    with pytest.raises(model_chat.ModelChatError) as caught:
        with model_chat.open_model_chat(
            case.startup, case.policy, cwd=tmp_path, environment={}
        ):
            pytest.fail("startup failure yielded")
    assert caught.value.cleanup_lease is owner


@pytest.mark.parametrize(
    "name", ["MODAL_TOKEN_ID", "HF_TOKEN", "HTTP_PROXY", "PYTHONPATH"]
)
def test_credentials_and_import_injection_rejected_before_start(case, tmp_path, name):
    with pytest.raises(ValueError):
        with model_chat.open_model_chat(
            case.startup, case.policy, cwd=tmp_path, environment={name: "private"}
        ):
            pytest.fail("unsafe environment started")
    assert case.captured["starts"] == []
