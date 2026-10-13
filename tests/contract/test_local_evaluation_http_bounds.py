"""Host-selected HTTP limits through real client transport and public reports."""
import hashlib
import json

import pytest

from Evaluator.base_client import _bounded_json_bytes
from Evaluator.client_factory import create_client
from Evaluator.config import VLLMSettings
from Evaluator.vllm_client import VLLMError
from synaptic_tuner.api.v1.reference.evaluation import LocalHttpEvaluatorBackendV1
from synaptic_tuner.api.v1 import EvaluationResultRequest
from test_reference_evaluation_v1 import _host, _run, _artifact_bytes, MODEL


class Response:
    status_code = 200

    def __init__(self, raw):
        self.raw, self.closed = raw, False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        # Deliberately split UTF-8 codepoints across provider chunks.
        for index in range(0, len(self.raw), 7):
            yield self.raw[index:index + 7]

    def close(self):
        self.closed = True


class Session:
    def __init__(self, response):
        self.response, self.closed, self.calls = response, False, []

    def request(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response

    def close(self):
        self.closed = True


def wire(monkeypatch, text):
    raw = json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}]},
                     ensure_ascii=False).encode()
    response = Response(raw)
    session = Session(response)
    monkeypatch.setattr("Evaluator.base_client.requests.Session", lambda: session)
    return raw, response, session


@pytest.mark.parametrize("field", ["max_request_bytes", "max_response_bytes"])
@pytest.mark.parametrize("value", [None, True, 0, -1, 1.5, 64 * 1024 * 1024 + 1])
def test_host_limits_reject_before_client_or_secrets(field, value):
    with pytest.raises(ValueError):
        LocalHttpEvaluatorBackendV1("vllm", host="localhost", port=8000, **{field: value})


def test_host_defaults_forward_finite_body_limits():
    client = LocalHttpEvaluatorBackendV1("vllm", host="localhost", port=8000).open(MODEL)
    assert client.max_request_bytes == client.max_response_bytes == 1 << 20


def test_complete_utf8_exact_response_bound_and_cleanup(monkeypatch):
    text = "界🙂" * 200
    raw, response, session = wire(monkeypatch, text)
    backend = LocalHttpEvaluatorBackendV1("vllm", host="localhost", port=8000,
                                         retries=0, max_response_bytes=len(raw))
    assert backend.open(MODEL).chat([{"role": "user", "content": "你好"}]).message == text
    assert response.closed and session.closed
    assert session.calls[0][1]["stream"] is True
    _, response, session = wire(monkeypatch, text)
    client = LocalHttpEvaluatorBackendV1("vllm", host="localhost", port=8000,
                                        retries=0, max_response_bytes=len(raw) - 1).open(MODEL)
    with pytest.raises(VLLMError):
        client.chat([{"role": "user", "content": "你好"}])
    assert response.closed and session.closed


def test_exact_request_bound_null_tokens_and_oversize_before_http(monkeypatch):
    settings = VLLMSettings(model="fixture", host="localhost", port=8000, max_tokens=None)
    messages = [{"role": "user", "content": "界🙂" * 100}]
    probe = create_client("vllm", settings)
    payload = probe._build_payload(messages)
    assert "max_tokens" not in payload
    body = _bounded_json_bytes(payload, 1 << 20)
    _, response, session = wire(monkeypatch, "ok")
    client = create_client("vllm", settings, retries=0, max_request_bytes=len(body), max_response_bytes=4096)
    assert client.chat(messages).message == "ok"
    assert session.calls[0][1]["data"] == body
    assert response.closed and session.closed
    monkeypatch.setattr("Evaluator.base_client.requests.Session", lambda: pytest.fail("oversize opened HTTP"))
    with pytest.raises(ValueError):
        create_client("vllm", settings, retries=3, max_request_bytes=len(body) - 1).chat(messages)


def test_http_complete_text_survives_public_artifact_readback(tmp_path, monkeypatch):
    text = "界🙂" * 200
    raw, response, session = wire(monkeypatch, text)
    backend = LocalHttpEvaluatorBackendV1("vllm", host="localhost", port=8000,
                                         retries=0, max_response_bytes=len(raw))
    _, api, _, _, _, sink = _host(tmp_path, backends={"local": backend})
    _, _, started = _run(api)
    result = api.evaluation.result(EvaluationResultRequest(started.run))
    artifact = next(value for value in result.artifacts if value.role == "evaluation_results")
    retained = _artifact_bytes(sink, started.run, artifact.role)
    assert len(retained) == artifact.size_bytes and hashlib.sha256(retained).hexdigest() == artifact.sha256
    assert all(row["response_text"] == text for row in json.loads(retained)["records"])
    assert response.closed and session.closed


def test_factory_omitted_transport_options_preserve_legacy_constructor(monkeypatch):
    from Evaluator import client_factory
    from Evaluator.enums import BackendType
    seen = []

    def constructor(*, settings, timeout, retries):
        seen.append((settings, timeout, retries))
        return object()

    monkeypatch.setitem(client_factory._CLIENT_REGISTRY, BackendType.VLLM, constructor)
    settings = VLLMSettings(model="fixture")
    create_client("vllm", settings, timeout=12, retries=1)
    assert seen == [(settings, 12, 1)]
