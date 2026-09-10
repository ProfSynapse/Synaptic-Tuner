from __future__ import annotations

import json
import threading

import pytest

from Evaluator import base_client, vllm_runtime
from Evaluator.chat_session import ChatSessionError, ChatSessionPolicy
from Evaluator.local_run_chat import LocalVLLMRunChatRuntime
from tuner.inference.run_chat import open_run_chat
from tests.inference.test_run_chat import _case


class _Process:
    def __init__(self) -> None:
        self.cleanup_pending = True
        self.calls = 0
        self.stopped = threading.Event()

    def close(self, **kwargs) -> bool:
        self.calls += 1
        self.cleanup_pending = False
        self.stopped.set()
        return True


@pytest.mark.parametrize("kind", ("full", "lora"))
def test_verified_run_to_local_vllm_chat_uses_one_owned_runtime(
    tmp_path, monkeypatch, kind
):
    runs, run, destination, preparer, _ = _case(tmp_path, kind)
    process = _Process()
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

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield json.dumps({"choices": [{"message": {"content": "hello"}}]}).encode()

        def close(self):
            return None

    class Session:
        def __init__(self):
            self.trust_env = True
            self.closed = False
            sessions.append(self)

        def request(self, *args, **kwargs):
            requests.append((args, kwargs))
            return Response()

        def close(self):
            self.closed = True

    monkeypatch.setattr(base_client.requests, "Session", Session)
    runtime = LocalVLLMRunChatRuntime(
        ChatSessionPolicy(1, 10, 20, 2, 4096),
        tmp_path,
        {},
        destination=destination,
        preparer=preparer,
        startup_options={
            "port": 9137,
            "python_executable": "/opt/inference/bin/python3",
        },
        max_request_bytes=8192,
    )

    with open_run_chat(
        runs,
        run,
        runtime=runtime,
    ) as opened:
        assert requests == []
        assert opened.local_model.model_kind == kind
        assert opened.model.model_kind == kind
        attempt = opened.local_model.root / opened.local_model.attempt
        assert attempt.is_dir()
        assert opened.session.chat("Hi").message == "hello"

    assert len(spawned) == 1 and len(requests) == 1
    argv, spawn_kwargs = spawned[0]
    assert argv[0] == "/opt/inference/bin/python3"
    assert argv[argv.index("--model") + 1] == str(
        tmp_path / "base" / "model" / "snapshots" / ("c" * 40)
        if preparer is not None
        else opened.local_model.model_path
    )
    if preparer is not None:
        assert preparer.calls == [("example/model", "c" * 40)]
    assert argv[argv.index("--tokenizer") + 1] == str(opened.local_model.tokenizer_path)
    assert spawn_kwargs["environment"]["HF_HUB_OFFLINE"] == "1"
    assert type(requests[0][1]["data"]) is bytes
    assert len(requests[0][1]["data"]) <= 8192
    payload = json.loads(requests[0][1]["data"])
    assert payload["model"] == "trained"
    assert payload["messages"] == [{"role": "user", "content": "Hi"}]
    assert requests[0][1]["headers"] == {"Content-Type": "application/json"}
    assert requests[0][1]["allow_redirects"] is False
    assert sessions[0].trust_env is False and sessions[0].closed
    assert process.stopped.wait(1) and process.calls == 1
    assert attempt.is_dir()


def test_backend_failure_closes_runtime_once_and_retains_materialization(
    tmp_path, monkeypatch
):
    runs, run, destination, preparer, _ = _case(tmp_path, "lora")
    process = _Process()
    spawned, requests, sessions = [], [], []

    monkeypatch.setattr(vllm_runtime, "_port_available", lambda *args: True)
    monkeypatch.setattr(
        vllm_runtime,
        "_spawn",
        lambda argv, **kwargs: spawned.append((argv, kwargs)) or process,
    )
    monkeypatch.setattr(vllm_runtime, "_leader_alive", lambda value: value is process)
    monkeypatch.setattr(vllm_runtime, "_ready", lambda *args: True)

    class Response:
        status_code = 200
        closed = False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            raise ValueError("private backend detail")
            yield b""

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
    runtime = LocalVLLMRunChatRuntime(
        ChatSessionPolicy(1, 10, 20, 2, 4096),
        tmp_path,
        {},
        destination=destination,
        preparer=preparer,
    )

    with pytest.raises(ChatSessionError) as caught:
        with open_run_chat(
            runs,
            run,
            runtime=runtime,
        ) as opened:
            attempt = opened.local_model.root / opened.local_model.attempt
            opened.session.chat("Hi")

    assert "private backend detail" not in str(caught.value)
    assert len(spawned) == 1 and len(requests) == 1
    assert process.stopped.wait(1) and process.calls == 1
    assert sessions[0].response.closed and sessions[0].closed
    assert attempt.is_dir()
