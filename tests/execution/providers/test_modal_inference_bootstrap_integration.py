"""Real bootstrap-to-chat composition with only external effects replaced."""

import json

import pytest

from Evaluator import base_client, vllm_runtime
from tests.evaluator.test_run_chat_local_integration import _Process
from tests.execution.providers.modal_inference_worker_fixtures import (
    mounted_launch_case,
)
from tuner.execution.providers.modal import inference_bootstrap as bootstrap


@pytest.mark.parametrize("kind", ("full", "lora"))
@pytest.mark.parametrize("failure", (False, True))
def test_signed_launch_reaches_real_chat_and_retains_model(
    tmp_path, monkeypatch, kind, failure
):
    case = mounted_launch_case(tmp_path, monkeypatch, model_kind=kind)
    checks, spawns, requests, clients = [], [], [], []
    process = _Process()
    monkeypatch.setattr(
        bootstrap,
        "verify_modal_inference_runtime",
        lambda config: checks.append(config.canonical_bytes),
    )
    monkeypatch.setattr(vllm_runtime, "_port_available", lambda *args: True)
    monkeypatch.setattr(
        vllm_runtime,
        "_spawn",
        lambda argv, **kwargs: spawns.append((argv, kwargs)) or process,
    )
    monkeypatch.setattr(vllm_runtime, "_leader_alive", lambda value: value is process)
    monkeypatch.setattr(vllm_runtime, "_ready", lambda *args: True)

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b'{"choices":[{"message":{"content":"hello"}}]}'

        def close(self):
            pass

    class Client:
        def __init__(self):
            self.trust_env = True
            self.closed = False
            clients.append(self)

        def request(self, *args, **kwargs):
            requests.append(kwargs)
            if failure:
                raise RuntimeError("synthetic backend failure")
            return Response()

        def close(self):
            self.closed = True

    monkeypatch.setattr(base_client.requests, "Session", Client)
    with bootstrap.open_modal_chat_worker(
        case.envelope.argument_bytes,
        **case.kwargs,
        cwd=tmp_path,
        environment={},
    ) as session:
        assert requests == []
        if failure:
            from Evaluator.chat_session import ChatSessionError

            with pytest.raises(ChatSessionError):
                session.chat("Hi")
        else:
            assert session.chat("Hi").message == "hello"
    assert len(checks) == 2
    assert all(
        value == case.kwargs["expectation"].configuration_bytes for value in checks
    )
    assert len(spawns) == len(requests) == 1
    argv, spawn = spawns[0]
    assert argv[0] == "/opt/conda/bin/python3"
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.73"
    assert spawn["environment"]["HF_HUB_OFFLINE"] == "1"
    assert "HF_TOKEN" not in spawn["environment"]
    payload = json.loads(requests[0]["data"])
    assert payload["model"] == "fixture-chat"
    assert payload["messages"] == [{"role": "user", "content": "Hi"}]
    assert (payload["max_tokens"], payload["temperature"], payload["top_p"]) == (
        73,
        0.25,
        0.875,
    )
    assert clients[0].closed and clients[0].trust_env is False
    assert process.stopped.wait(1) and not process.cleanup_pending
    # Preparation artifacts persist independently of session success/cleanup.
    model = argv[argv.index("--model") + 1]
    from pathlib import Path

    assert Path(model).is_dir()
