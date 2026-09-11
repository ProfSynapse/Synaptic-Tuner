from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import threading
from types import SimpleNamespace

import pytest

from Evaluator import verified_vllm_chat as composition
from Evaluator.chat_session import ChatSessionError, ChatSessionPolicy
from Evaluator.protocols import BackendResponse
from Evaluator.vllm_runtime import (
    ExplicitNetworkVLLMSource,
    VerifiedLocalVLLMSource,
    VLLMRuntimeLease,
    VLLMStartupSpec,
)


class Process:
    def __init__(self, *, closes=True):
        self.cleanup_pending = True
        self.calls = 0
        self.closes = closes

    def close(self, **kwargs):
        self.calls += 1
        self.cleanup_pending = not self.closes
        return self.closes


@pytest.fixture
def setup(tmp_path, monkeypatch):
    process = Process()
    runtime = VLLMRuntimeLease(
        process, host="127.0.0.1", port=9137, served_model_name="trained"
    )
    captured = {"starts": [], "requests": []}

    def start(spec, **kwargs):
        captured["starts"].append((spec, kwargs))
        return runtime

    class Client:
        def __init__(self, settings, **kwargs):
            captured["settings"] = settings
            captured["client"] = kwargs

        def chat(self, messages):
            captured["requests"].append(messages)
            return BackendResponse(message="hello", raw={}, latency_s=0.01)

    monkeypatch.setattr(composition, "start_vllm_runtime", start)
    monkeypatch.setattr(composition, "VLLMClient", Client)
    # This fixture qualifies composition, not target authentication. The real
    # startup/ServingTarget integration is separately tested below.
    spec = VLLMStartupSpec(
        VerifiedLocalVLLMSource(object()), served_model_name="trained"
    )
    policy = ChatSessionPolicy(1, 10, 20, 4, 4096)
    return spec, policy, process, runtime, captured


def test_composition_uses_one_runtime_client_and_explicit_prompt(
    setup, tmp_path, monkeypatch
):
    spec, policy, process, runtime, captured = setup
    monkeypatch.setenv("HF_TOKEN", "fixture-not-a-credential")
    monkeypatch.setenv("VLLM_API_KEY", "fixture-not-a-credential")
    with composition.verified_vllm_chat(
        spec, policy, cwd=tmp_path, environment={}
    ) as chat:
        assert len(captured["starts"]) == 1
        assert captured["requests"] == []
        settings = captured["settings"]
        assert (settings.host, settings.port, settings.model) == (
            "127.0.0.1",
            9137,
            "trained",
        )
        assert settings.api_key is None
        assert settings.scheme == "http"
        assert settings.model_path is None and settings.lora_adapter is None
        assert captured["client"] == {
            "timeout": 1,
            "retries": 0,
            "trust_environment": False,
            "allow_redirects": False,
            "max_request_bytes": 1 << 20,
            "max_response_bytes": 1 << 20,
        }
        assert chat.chat("Hi").message == "hello"
        assert len(captured["requests"]) == 1
    assert process.calls == 1
    assert not runtime.cleanup_pending


def test_same_absolute_deadline_survives_startup(setup, tmp_path, monkeypatch):
    spec, policy, process, runtime, captured = setup
    now = [100.0]
    start = composition.start_vllm_runtime

    def slow_start(*args, **kwargs):
        now[0] += 0.2
        return start(*args, **kwargs)

    monkeypatch.setattr(composition, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(composition, "start_vllm_runtime", slow_start)
    with composition.verified_vllm_chat(
        spec, policy, cwd=tmp_path, environment={}, deadline=100.25
    ) as chat:
        assert captured["starts"][0][1]["deadline"] == 100.25
        assert chat._absolute_deadline == 100.25
        assert captured["requests"] == []
        now[0] = 100.25
        with pytest.raises(ChatSessionError):
            chat.chat("late")
    assert process.calls == 1


@pytest.mark.parametrize("phase", ("startup", "client", "session"))
def test_deadline_expiry_during_construction_never_yields(
    setup, tmp_path, monkeypatch, phase
):
    spec, policy, process, runtime, captured = setup
    now = [100.0]
    monkeypatch.setattr(composition, "time", SimpleNamespace(monotonic=lambda: now[0]))
    name = {
        "startup": "start_vllm_runtime",
        "client": "VLLMClient",
        "session": "ChatSession",
    }[phase]
    original = getattr(composition, name)

    def slow(*args, **kwargs):
        value = original(*args, **kwargs)
        now[0] = 101.0
        return value

    monkeypatch.setattr(composition, name, slow)
    with pytest.raises(composition.VerifiedVLLMChatError):
        with composition.verified_vllm_chat(
            spec, policy, cwd=tmp_path, environment={}, deadline=101.0
        ):
            pytest.fail("yielded after absolute deadline")
    assert process.calls == 1 and captured["requests"] == []
    if phase == "startup":
        assert "client" not in captured


@pytest.mark.parametrize("deadline", (True, "100", float("nan"), float("inf")))
def test_invalid_absolute_deadline_is_rejected_before_startup(
    setup, tmp_path, deadline
):
    spec, policy, _, _, captured = setup
    with pytest.raises((TypeError, ValueError)):
        with composition.verified_vllm_chat(
            spec, policy, cwd=tmp_path, environment={}, deadline=deadline
        ):
            pytest.fail("invalid deadline accepted")
    assert captured["starts"] == []


@pytest.mark.parametrize("point", ["settings", "client", "session"])
@pytest.mark.parametrize("interrupt", [False, True])
def test_construction_failure_closes_owner_and_preserves_interrupt(
    setup, tmp_path, monkeypatch, point, interrupt
):
    spec, policy, process, runtime, _ = setup
    failure = KeyboardInterrupt() if interrupt else RuntimeError("private-detail")

    def fail(*args, **kwargs):
        raise failure

    symbol = {
        "settings": "VLLMSettings",
        "client": "VLLMClient",
        "session": "ChatSession",
    }[point]
    monkeypatch.setattr(composition, symbol, fail)
    expected = KeyboardInterrupt if interrupt else composition.VerifiedVLLMChatError
    with pytest.raises(expected) as caught:
        with composition.verified_vllm_chat(spec, policy, cwd=tmp_path, environment={}):
            pytest.fail("yielded after construction failure")
    if interrupt:
        assert caught.value is failure
    else:
        assert "private-detail" not in str(caught.value)
        assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert process.calls == 1
    assert not runtime.cleanup_pending


def test_unresolved_construction_cleanup_retains_exact_lease(
    setup, tmp_path, monkeypatch
):
    spec, policy, process, runtime, _ = setup
    process.closes = False

    def fail(*args, **kwargs):
        raise RuntimeError("private-detail")

    monkeypatch.setattr(composition, "VLLMClient", fail)
    with pytest.raises(composition.VerifiedVLLMChatError) as caught:
        with composition.verified_vllm_chat(spec, policy, cwd=tmp_path, environment={}):
            pytest.fail("yielded")
    assert caught.value.cleanup_lease is runtime
    assert runtime.cleanup_pending
    process.closes = True
    assert caught.value.cleanup_lease.close()


def test_user_exception_is_not_reclassified(setup, tmp_path):
    spec, policy, process, _, _ = setup
    failure = LookupError("caller")
    with pytest.raises(LookupError) as caught:
        with composition.verified_vllm_chat(spec, policy, cwd=tmp_path, environment={}):
            raise failure
    assert caught.value is failure
    assert process.calls == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tokens": True},
        {"max_tokens": 0},
        {"max_tokens": 32769},
        {"temperature": float("nan")},
        {"temperature": -1},
        {"top_p": 0},
        {"max_response_bytes": True},
        {"max_response_bytes": 0},
        {"max_request_bytes": True},
        {"max_request_bytes": 0},
        {"max_request_bytes": -1},
        {"max_request_bytes": None},
        {"max_request_bytes": 64 * 1024 * 1024 + 1},
    ],
)
def test_invalid_generation_is_rejected_before_startup(setup, tmp_path, kwargs):
    spec, policy, _, _, captured = setup
    with pytest.raises(ValueError):
        with composition.verified_vllm_chat(
            spec, policy, cwd=tmp_path, environment={}, **kwargs
        ):
            pytest.fail("yielded")
    assert captured["starts"] == []


def test_request_limit_reaches_real_client_and_closes_owned_runtime(
    setup, tmp_path, monkeypatch
):
    from Evaluator import base_client
    from Evaluator.chat_session import ChatSessionError
    from Evaluator.vllm_client import VLLMClient

    spec, policy, process, runtime, captured = setup
    attempts = []

    def forbidden_session():
        attempts.append("session")
        raise AssertionError("oversized request reached HTTP")

    monkeypatch.setattr(composition, "VLLMClient", VLLMClient)
    monkeypatch.setattr(base_client.requests, "Session", forbidden_session)
    with composition.verified_vllm_chat(
        spec, policy, cwd=tmp_path, environment={}, max_request_bytes=32
    ) as chat:
        assert len(captured["starts"]) == 1
        with pytest.raises(ChatSessionError):
            chat.chat("Hi")
    assert attempts == []
    assert process.calls == 1 and not runtime.cleanup_pending


def test_explicit_network_source_is_not_relabeled_verified(setup, tmp_path):
    spec, policy, _, _, captured = setup
    spec = replace(spec, source=ExplicitNetworkVLLMSource("fixture/model"))
    with pytest.raises(TypeError):
        with composition.verified_vllm_chat(spec, policy, cwd=tmp_path, environment={}):
            pytest.fail("yielded")
    assert captured["starts"] == []


def _real_target(tmp_path, kind):
    from synaptic_tuner.api.v1.runs_facade import RunsAPI
    from tests.inference.test_retrieved_model import _fixture, _safe, _tar
    from tests.inference.test_serving_target import _snapshot
    from tuner.inference.retrieved_model import materialize_verified_sft_model
    from tuner.inference.serving_target import ServingTarget

    root = tmp_path / "retrieved"
    root.mkdir()
    run, values, operations = _fixture(root)
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
    retrieved = materialize_verified_sft_model(RunsAPI(operations), run, root)
    return ServingTarget(retrieved, _snapshot(tmp_path) if kind == "lora" else None)


@pytest.mark.parametrize("kind", ["full", "lora"])
@pytest.mark.parametrize("outcome", ["success", "error", "interrupt", "timeout"])
@pytest.mark.parametrize("with_deadline", [False, True])
def test_real_target_runtime_client_session_chain_with_fake_effects(
    tmp_path, monkeypatch, kind, outcome, with_deadline
):
    from Evaluator import base_client, vllm_runtime
    from Evaluator.chat_session import ChatSessionError

    target = _real_target(tmp_path, kind)
    stopped = threading.Event()
    process = Process()
    original_close = process.close

    def close(**kwargs):
        stopped.set()
        return original_close(**kwargs)

    process.close = close
    spawned, requests, sessions = [], [], []
    monkeypatch.setattr(vllm_runtime, "_port_available", lambda *args: True)
    monkeypatch.setattr(
        vllm_runtime,
        "_spawn",
        lambda argv, **kwargs: spawned.append((argv, kwargs)) or process,
    )
    monkeypatch.setattr(vllm_runtime, "_leader_alive", lambda value: value is process)
    expected_names = ("synaptic-base", "trained") if kind == "lora" else ("trained",)
    monkeypatch.setattr(
        vllm_runtime,
        "_ready",
        lambda host, port, names, timeout: names == expected_names,
    )

    class Response:
        status_code = 200
        closed = False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            if outcome == "interrupt":
                raise KeyboardInterrupt()
            if outcome == "error":
                raise ValueError("private backend detail")
            if outcome == "timeout":
                assert stopped.wait(2), "cleanup never started"
            yield json.dumps({"choices": [{"message": {"content": "hello"}}]}).encode()

        def close(self):
            self.closed = True

    class Session:
        def __init__(self):
            self.trust_env = True
            self.closed = False
            self.response = Response()
            sessions.append(self)

        def request(self, *args, **kwargs):
            requests.append((args, kwargs))
            return self.response

        def close(self):
            self.closed = True

    monkeypatch.setattr(base_client.requests, "Session", Session)
    spec = VLLMStartupSpec(VerifiedLocalVLLMSource(target), served_model_name="trained")
    policy = ChatSessionPolicy(0.05 if outcome == "timeout" else 1, 10, 20, 2, 4096)
    deadline = composition.time.monotonic() + 10 if with_deadline else None

    def converse():
        with composition.verified_vllm_chat(
            spec, policy, cwd=tmp_path, environment={}, deadline=deadline
        ) as chat:
            assert requests == []
            if with_deadline:
                assert chat._absolute_deadline == deadline
            assert chat.chat("Hi").message == "hello"

    if outcome == "success":
        converse()
    else:
        with pytest.raises(
            KeyboardInterrupt if outcome == "interrupt" else ChatSessionError
        ):
            converse()
    assert len(spawned) == 1 and len(requests) == 1
    argv, kwargs = spawned[0]
    assert kwargs["environment"]["HF_HUB_OFFLINE"] == "1"
    assert argv[argv.index("--model") + 1] == str(
        target.base_model_path or target.model_path
    )
    assert argv[argv.index("--tokenizer") + 1] == str(target.tokenizer_path)
    assert json.loads(requests[0][1]["data"])["model"] == "trained"
    assert requests[0][1]["headers"] == {"Content-Type": "application/json"}
    assert requests[0][1]["allow_redirects"] is False
    assert not sessions[0].trust_env
    assert stopped.wait(1) and process.calls == 1
