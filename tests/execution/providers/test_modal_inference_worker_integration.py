"""Mounted chat preparation reuses the actual Modal pinned-model adapter."""

from dataclasses import fields
import json

import pytest

from Evaluator import base_client, vllm_runtime
from Evaluator.verified_vllm_chat import verified_vllm_chat
from tests.evaluator.test_run_chat_local_integration import _Process
from tests.execution.providers import test_modal_inference_preparation as config_cases

from tests.execution.providers.modal_inference_worker_fixtures import (
    mounted_launch_case,
)
from tuner.execution.providers.modal import inference_model, inference_worker
from tuner.execution.providers.modal.inference_model import ModalPinnedModelPreparer
from tuner.execution.providers.modal.inference_worker import prepare_modal_chat_worker
from tuner.execution.foundation_v2.canonical import parse_canonical_object
from tuner.inference.serving_target import ServingTarget


@pytest.mark.parametrize("model_kind", ("full", "lora"))
def test_signed_mounted_bytes_reach_existing_modal_model_preparer(
    tmp_path, monkeypatch, model_kind
):
    case = mounted_launch_case(tmp_path, monkeypatch, model_kind=model_kind)
    persistent = case.roots["cache"] / "snapshots"
    destination = case.roots["destination"] / "base-models"
    scratch = case.roots["destination"] / "base-scratch"
    for root in (persistent, destination, scratch):
        root.mkdir()
    calls = []

    def prepare_snapshot(**kwargs):
        # Only external model acquisition is replaced. Adapter root checks,
        # snapshot capture, target validation and the whole artifact path run.
        calls.append(kwargs)
        path = destination / "model" / "snapshots" / kwargs["revision"]
        path.mkdir(parents=True)
        (path / "config.json").write_bytes(b'{"model_type":"fixture"}')
        (path / "weights.bin").write_bytes(b"fixture-weights")
        return path

    monkeypatch.setattr(inference_model, "prepare_model_snapshot", prepare_snapshot)
    preparer = ModalPinnedModelPreparer(
        persistent_root=persistent,
        destination_root=destination,
        scratch_root=scratch,
        token=None,
    )
    prepared = prepare_modal_chat_worker(
        case.envelope.argument_bytes, **(case.kwargs | {"preparer": preparer})
    )
    target = prepared.startup.source.target
    assert type(target) is ServingTarget
    target.validate()
    assert target.retrieved.run.to_dict() == case.source["run"]
    assert tuple(item.to_dict() for item in target.retrieved.artifacts) == tuple(
        case.source["artifacts"]
    )
    assert target.retrieved.model_kind == model_kind
    assert target.retrieved.model_ref == case.workload["model_ref"]
    assert target.retrieved.model_revision == case.workload["model_revision"]
    assert target.retrieved.tokenizer_revision == case.workload["tokenizer_revision"]
    if model_kind == "full":
        assert calls == []
        assert target.base_snapshot is None
    else:
        assert len(calls) == 1
        assert calls[0] == dict(
            model_ref=case.workload["model_ref"],
            revision=case.workload["model_revision"],
            token=None,
            persistent_root=persistent,
            destination_root=destination,
            scratch_root=scratch,
        )
        assert target.base_snapshot.model_ref == case.workload["model_ref"]
        assert target.base_snapshot.revision == case.workload["model_revision"]
    assert case.preparer.calls == []


@pytest.mark.parametrize("model_kind", ("full", "lora"))
def test_admitted_settings_reach_existing_runtime_and_http_client(
    tmp_path, monkeypatch, model_kind
):
    case = mounted_launch_case(tmp_path, monkeypatch, model_kind=model_kind)
    supplied, admissions = [], []
    startup_type = vllm_runtime.VLLMStartupSpec
    admit = inference_worker.admit_modal_chat_launch

    def admission(*args, **kwargs):
        result = admit(*args, **kwargs)
        admissions.append(result)
        return result

    def startup(**kwargs):
        supplied.append(kwargs)
        return startup_type(**kwargs)

    monkeypatch.setattr(inference_worker, "VLLMStartupSpec", startup)
    monkeypatch.setattr(inference_worker, "admit_modal_chat_launch", admission)
    prepared = prepare_modal_chat_worker(case.envelope.argument_bytes, **case.kwargs)
    assert set(supplied[0]) == {field.name for field in fields(startup_type)}
    assert prepared.admission.argument_bytes == case.envelope.argument_bytes
    assert prepared.admission is not admissions[0]
    assert prepared.admission.configuration.canonical_bytes == (
        case.kwargs["expectation"].configuration_bytes
    )
    assert supplied[0] == dict(
        source=prepared.startup.source,
        served_model_name="fixture-chat",
        host="127.0.0.1",
        port=8000,
        gpu_memory_utilization=0.73,
        tensor_parallel_size=1,
        enforce_eager=False,
        tokenizer_mode="mistral",
        max_lora_rank=32,
        startup_timeout_s=600,
        readiness_request_timeout_s=0.75,
        python_executable="/opt/conda/bin/python3",
    )
    assert vars_policy(prepared.configured_policy) == (60, 300, 900, 32, 65536)
    assert (prepared.max_tokens, prepared.temperature, prepared.top_p) == (
        73,
        0.25,
        0.875,
    )
    assert (prepared.max_request_bytes, prepared.max_response_bytes) == (8192, 65536)

    process = _Process()
    spawned, requests, sessions, probes = [], [], [], []
    monkeypatch.setattr(vllm_runtime, "_port_available", lambda *args: True)
    monkeypatch.setattr(
        vllm_runtime,
        "_spawn",
        lambda argv, **kwargs: spawned.append((argv, kwargs)) or process,
    )
    monkeypatch.setattr(vllm_runtime, "_leader_alive", lambda value: value is process)
    monkeypatch.setattr(
        vllm_runtime,
        "_ready",
        lambda *args: probes.append(args) or True,
    )

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b'{"choices":[{"message":{"content":"hello"}}]}'

        def close(self):
            pass

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
    # This deliberately exercises projection only with fake effects. Production
    # bootstrap still needs runtime-lock admission and remaining-deadline policy.
    with verified_vllm_chat(
        prepared.startup,
        prepared.configured_policy,
        cwd=tmp_path,
        environment={},
        max_tokens=prepared.max_tokens,
        temperature=prepared.temperature,
        top_p=prepared.top_p,
        max_request_bytes=prepared.max_request_bytes,
        max_response_bytes=prepared.max_response_bytes,
    ) as session:
        assert requests == []
        assert session.chat("Hi").message == "hello"
    assert len(spawned) == len(requests) == 1
    argv, spawn = spawned[0]
    assert argv[0] == "/opt/conda/bin/python3"
    assert "--enforce-eager" not in argv
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.73"
    assert argv[argv.index("--tokenizer-mode") + 1] == "mistral"
    assert probes[0][3] == 0.75
    assert probes[0][2] == (
        ("fixture-chat", "synaptic-base") if model_kind == "lora" else ("fixture-chat",)
    )
    assert spawn["environment"]["HF_HUB_OFFLINE"] == "1"
    assert "HF_TOKEN" not in spawn["environment"]
    if model_kind == "lora":
        assert argv[argv.index("--max-lora-rank") + 1] == "32"
    payload = json.loads(requests[0][1]["data"])
    assert payload["model"] == "fixture-chat"
    assert payload["max_tokens"] == 73
    assert payload["temperature"] == 0.25 and payload["top_p"] == 0.875
    assert payload["messages"] == [{"role": "user", "content": "Hi"}]
    assert len(requests[0][1]["data"]) <= 8192
    assert sessions[0].closed and sessions[0].trust_env is False
    assert process.stopped.wait(1) and process.calls == 1
    assert prepared.startup.source.target.retrieved.model_path.is_dir()


def vars_policy(policy):
    return tuple(getattr(policy, field.name) for field in fields(policy))


def test_maximum_admitted_alias_reaches_real_lora_projection(tmp_path, monkeypatch):
    document = config_cases._document

    def configured(*args, **kwargs):
        value = document(*args, **kwargs)
        value["serving"]["served_model_name"] = "a" * 96
        return value

    monkeypatch.setattr(config_cases, "_document", configured)
    case = mounted_launch_case(tmp_path, monkeypatch, model_kind="lora")
    prepared = prepare_modal_chat_worker(case.envelope.argument_bytes, **case.kwargs)
    projection = vllm_runtime._projection(
        prepared.startup, cwd=tmp_path, environment={}
    )
    assert projection.expected_model_names == ("a" * 96, "synaptic-base")
    assert projection.argv[projection.argv.index("--lora-modules") + 1].startswith(
        "a" * 96 + "="
    )


@pytest.mark.parametrize(
    "phase", ("expired", "expectation", "admission", "model", "interrupt", "exit")
)
def test_final_clock_cannot_renew_or_mutate_preparation(tmp_path, monkeypatch, phase):
    case = mounted_launch_case(tmp_path, monkeypatch)
    claim = parse_canonical_object(case.envelope.claim, name="fixture claim")
    retained = {}
    admit = inference_worker.admit_modal_chat_launch
    prepare = inference_worker.prepare_serving_target

    def admission(*args, **kwargs):
        result = admit(*args, **kwargs)
        retained["admission"] = result
        return result

    def target(*args, **kwargs):
        result = prepare(*args, **kwargs)
        retained["target"] = result
        return result

    monkeypatch.setattr(inference_worker, "admit_modal_chat_launch", admission)
    monkeypatch.setattr(inference_worker, "prepare_serving_target", target)

    class Clock:
        calls = 0

        def now_iso(self):
            self.calls += 1
            if self.calls == 2:
                if phase == "expired":
                    return claim["expires_at"]
                if phase == "interrupt":
                    raise KeyboardInterrupt()
                if phase == "exit":
                    raise SystemExit()
                if phase == "expectation":
                    object.__setattr__(
                        case.kwargs["expectation"], "artifact_root", "/other"
                    )
                if phase == "admission":
                    object.__setattr__(retained["admission"], "_claim", b"changed")
                if phase == "model":
                    object.__setattr__(
                        retained["target"].retrieved, "model_ref", "changed"
                    )
            return "2026-09-09T12:02:01Z"

    clock = Clock()
    error = {
        "interrupt": KeyboardInterrupt,
        "exit": SystemExit,
    }.get(phase, inference_worker.ModalInferenceWorkerError)
    with pytest.raises(error):
        prepare_modal_chat_worker(
            case.envelope.argument_bytes, **(case.kwargs | {"clock": clock})
        )
    assert clock.calls == 2
    if phase == "expired":
        assert retained["target"].retrieved.model_path.is_dir()


@pytest.mark.parametrize("failure", (RuntimeError, KeyboardInterrupt, SystemExit))
def test_owned_descriptor_cleanup_attempts_every_close(monkeypatch, failure):
    calls = []

    def close(descriptor):
        calls.append(descriptor)
        if descriptor == 12:
            raise failure()

    monkeypatch.setattr(inference_worker.os, "close", close)
    expected = (
        inference_worker.ModalInferenceWorkerError
        if failure is RuntimeError
        else failure
    )
    with pytest.raises(expected) as caught:
        inference_worker._close_owned(11, 12, 13)
    assert calls == [11, 12, 13]
    if failure is RuntimeError:
        assert str(caught.value) == "modal_inference_worker_invalid"
        assert caught.value.__cause__ is None


@pytest.mark.parametrize("failure", (RuntimeError, KeyboardInterrupt, SystemExit))
def test_owned_descriptor_cleanup_preserves_active_failure(monkeypatch, failure):
    calls = []
    primary = ValueError("test-primary-failure")

    def close(descriptor):
        calls.append(descriptor)
        if descriptor == 12:
            raise failure()

    monkeypatch.setattr(inference_worker.os, "close", close)
    with pytest.raises(ValueError) as caught:
        try:
            raise primary
        finally:
            inference_worker._close_owned(11, 12, 13)
    assert caught.value is primary
    assert calls == [11, 12, 13]
