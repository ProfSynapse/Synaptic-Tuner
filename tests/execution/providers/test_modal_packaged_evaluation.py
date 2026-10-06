"""Training completion and signed same-job evaluation remain separate records."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from types import SimpleNamespace

import pytest

from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.providers.modal import packaged_worker as worker_module
from tuner.execution.providers.modal.packaged_dispatch import (
    build_modal_packaged_dispatch, parse_modal_packaged_dispatch,
)
from tuner.execution.providers.modal.packaged_reader import ModalPackagedReader
from tuner.execution.providers.modal.packaged_worker import ModalPackagedWorker, ModalPackagedWorkerRoots
from tuner.runtime import post_training_eval
from tuner.training.contracts import CanonicalDocument
from tuner.training.packaged_compilation import compile_packaged_sft_workload
from tests.execution.providers.test_modal_packaged_dispatch import Auth, _case
from tests.execution.providers.test_modal_packaged_worker import Executor, Signer
from tests.training.test_modal_post_training_compilation import _evaluation


def _record(config, *, workload_digest, adapter_digest, passed=True):
    cases = [{
        "id": scenario["id"], "status": "pass" if passed else "fail",
        "response": "hello" if passed else "", "latency_seconds": 0.01,
        "matched_path": "default" if passed else None, "error_code": None,
    } for scenario in config["evaluation"]["scenarios"]]
    return {
        "schema_version": "synaptic-post-training-evaluation/v1",
        "status": "completed", "gate_passed": passed,
        "failure_code": None if passed else "gate_failed",
        "case_count": len(cases), "passed_count": len(cases) if passed else 0,
        "pass_rate": 1.0 if passed else 0.0,
        "min_pass_rate": config["evaluation"]["min_pass_rate"],
        "cases": cases,
        "bindings": {"workload_digest": workload_digest,
                     "model_snapshot_digest": "a" * 64,
                     "adapter_digest": adapter_digest},
    }


class CallbackExecutor(Executor):
    def execute(self, **kwargs):
        callback = kwargs.pop("on_training_complete")
        result = super().execute(**kwargs)
        artifact = next(item for item in json.loads(result.inventory_path.read_bytes())["artifacts"]
                        if item["role"] == "final_model")
        context = SimpleNamespace(
            base_model_path=kwargs["paths"].cache,
            adapter_path=kwargs["paths"].state,
            tokenizer_path=kwargs["paths"].state,
            validate=lambda: None,
            environment={}, python_executable="/usr/local/bin/python",
            bindings={"workload_digest": kwargs["execution_binding"].workload_digest,
                      "model_snapshot_digest": "a" * 64,
                      "adapter_digest": artifact["sha256"]},
        )
        callback(result, context)
        return result


def _worker(tmp_path, monkeypatch, *, signer=None):
    binding, receipt, original_workload, policy = _case()
    auth, signer = Auth(), signer if signer is not None else Signer()
    original = parse_modal_packaged_dispatch(
        build_modal_packaged_dispatch(binding, receipt, original_workload, policy, auth,
                                      key_ref="dispatch-key", environment=(("PATH", "/usr/bin:/bin"),)),
        auth,
    )
    document = json.loads(original_workload)
    document["configuration"]["document"]["post_training"] = _evaluation()
    opted = compile_packaged_sft_workload(resolved_config=CanonicalDocument.from_mapping(
        document["configuration"]["document"])).canonical_bytes
    values = {item.name: getattr(original, item.name) for item in fields(original)}
    values["workload_bytes"] = opted
    values["submit_command"] = original.submit_command
    # The parser is separately covered by its signed-dispatch tests. Here the
    # supplied frame isolates the worker's phase ordering and publication.
    monkeypatch.setattr(worker_module, "parse_modal_packaged_dispatch",
                        lambda _raw, _verifier: SimpleNamespace(**values))
    roots = ModalPackagedWorkerRoots(
        (tmp_path / "control").resolve(), (tmp_path / "artifacts").resolve(),
        (tmp_path / "cache").resolve(),
    )
    for root in (roots.control, roots.artifacts, roots.cache):
        root.mkdir()
    staged = roots.artifacts / receipt.path
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"packaged-prepared-input")
    executor = CallbackExecutor()
    worker = ModalPackagedWorker(
        expected_facts=binding.provider_facts, dispatch_verifier=auth,
        trainer_executor=executor, evidence_signer=signer, roots=roots,
    )
    return binding, worker, executor, roots, opted


def test_worker_commits_training_before_evaluation_and_retains_record(tmp_path, monkeypatch):
    binding, worker, executor, roots, opted = _worker(tmp_path, monkeypatch)
    config = json.loads(opted)["configuration"]["document"]["post_training"]
    events = []

    def evaluate(_config, **kwargs):
        assert _config == config
        events.append("evaluate")
        assert (roots.control / f"operations/{binding.command.operation.effect.effect_id}/evidence/packaged-completion.json").is_file()
        return _record(config, workload_digest=kwargs["bindings"]["workload_digest"],
                       adapter_digest=kwargs["bindings"]["adapter_digest"])

    monkeypatch.setattr(post_training_eval, "execute_post_training_evaluation", evaluate)
    result = worker(b"dispatch", "fc-1", commit_artifacts=lambda: events.append("artifacts"),
                    commit_control=lambda: events.append("control"))
    assert result["status_code"] == "completed"
    assert events[:3] == ["artifacts", "control", "evaluate"]
    assert len(executor.calls) == 1
    effect = binding.command.operation.effect.effect_id
    evaluation = roots.artifacts / f"operations/{effect}/evaluation/record.json"
    assert evaluation.is_file()
    assert json.loads(evaluation.read_bytes())["evaluation"]["gate_passed"] is True


def test_worker_publishes_large_unicode_response_without_truncation(tmp_path, monkeypatch):
    binding, worker, _, roots, opted = _worker(tmp_path, monkeypatch)
    config = json.loads(opted)["configuration"]["document"]["post_training"]
    unit = "The moon rose over the sea. "
    prose = unit * ((post_training_eval.MAX_EVALUATION_RESPONSE_BYTES - 512) // len(unit)) + "🌙"
    assert post_training_eval.MAX_EVALUATION_RESPONSE_BYTES - 1024 < len(prose.encode("utf-8")) < post_training_eval.MAX_EVALUATION_RESPONSE_BYTES

    def evaluate(_config, **kwargs):
        record = _record(config, workload_digest=kwargs["bindings"]["workload_digest"],
                         adapter_digest=kwargs["bindings"]["adapter_digest"])
        record["cases"][0]["response"] = prose
        return record

    monkeypatch.setattr(post_training_eval, "execute_post_training_evaluation", evaluate)
    result = worker(b"dispatch", "fc-1", commit_artifacts=lambda: None,
                    commit_control=lambda: None)
    assert result["status_code"] == "completed"
    effect = binding.command.operation.effect.effect_id
    raw = (roots.artifacts / f"operations/{effect}/evaluation/record.json").read_bytes()
    assert post_training_eval.MAX_EVALUATION_RESPONSE_BYTES < len(raw) < post_training_eval.MAX_EVALUATION_RECORD_BYTES
    assert json.loads(raw)["evaluation"]["cases"][0]["response"] == prose


def test_worker_signed_large_evaluation_round_trips_through_reader(tmp_path, monkeypatch):
    class DigestAuth:
        def sign(self, purpose, payload, key_ref):
            return hashlib.sha256(payload).digest()

        def verify(self, purpose, payload, tag, key_ref):
            return (purpose == "modal-packaged-evaluation/v1"
                    and key_ref == "dispatch-key"
                    and tag == hashlib.sha256(payload).digest())

    auth = DigestAuth()
    binding, worker, _, roots, opted = _worker(tmp_path, monkeypatch, signer=auth)
    config = json.loads(opted)["configuration"]["document"]["post_training"]
    compiled = compile_packaged_sft_workload(
        resolved_config=CanonicalDocument.from_mapping(json.loads(opted)["configuration"]["document"]))
    unit = "A luminous harbor answered. "
    prose = unit * ((post_training_eval.MAX_EVALUATION_RESPONSE_BYTES - 512) // len(unit)) + "🌙"
    assert 64 * 1024 < len(prose.encode("utf-8")) < post_training_eval.MAX_EVALUATION_RESPONSE_BYTES
    retained = {}

    def evaluate(_config, **kwargs):
        retained["adapter_digest"] = kwargs["bindings"]["adapter_digest"]
        record = _record(config, workload_digest=compiled.fingerprint,
                         adapter_digest=retained["adapter_digest"])
        record["cases"][0]["response"] = prose
        return record

    monkeypatch.setattr(post_training_eval, "execute_post_training_evaluation", evaluate)
    assert worker(b"dispatch", "fc-1", commit_artifacts=lambda: None,
                  commit_control=lambda: None)["status_code"] == "completed"
    effect = binding.command.operation.effect.effect_id
    raw = (roots.artifacts / f"operations/{effect}/evaluation/record.json").read_bytes()
    tag = (roots.artifacts / f"operations/{effect}/evaluation/record.mac").read_bytes()
    completion_raw = (roots.control / f"operations/{effect}/evidence/packaged-completion.json").read_bytes()
    completion = SimpleNamespace(
        effect_id=effect, provider_job_ref="fc-1",
        completion_digest=hashlib.sha256(completion_raw).hexdigest(),
        members=(SimpleNamespace(role="final_model", sha256=retained["adapter_digest"]),),
    )
    # _worker changes only its in-memory workload to isolate worker publication;
    # project the matching compiled fingerprint for the reader's exact binding.
    reader_binding = SimpleNamespace(
        execution_binding=SimpleNamespace(workload_digest=compiled.fingerprint,
                                          binding_digest=binding.execution_binding.binding_digest),
        command_digest=binding.command_digest, provider_facts=binding.provider_facts,
    )
    reader = ModalPackagedReader.__new__(ModalPackagedReader)
    reader._facade, reader._verifier, reader._key_ref = ReadFacade(raw, tag), auth, "dispatch-key"
    monkeypatch.setattr(ModalPackagedReader, "observe_completion",
                        lambda self, _binding, *, provider_job_ref: completion)
    readback = reader.read_evaluation(reader_binding, provider_job_ref="fc-1",
                                      workload_bytes=opted)
    assert readback == raw
    assert len(readback) > 64 * 1024
    assert json.loads(readback)["evaluation"]["cases"][0]["response"] == prose


def test_worker_evaluation_failure_keeps_training_completion(tmp_path, monkeypatch):
    binding, worker, executor, roots, _ = _worker(tmp_path, monkeypatch)
    events = []

    def fail(*_args, **_kwargs):
        events.append("evaluate")
        raise RuntimeError("private evaluation failure")

    monkeypatch.setattr(post_training_eval, "execute_post_training_evaluation", fail)
    result = worker(b"dispatch", "fc-1", commit_artifacts=lambda: events.append("artifacts"),
                    commit_control=lambda: events.append("control"))
    assert result["status_code"] == "failed"
    assert events == ["artifacts", "control", "evaluate"]
    assert len(executor.calls) == 1
    effect = binding.command.operation.effect.effect_id
    assert (roots.control / f"operations/{effect}/evidence/packaged-completion.json").is_file()
    assert len(tuple((roots.artifacts / f"operations/{effect}/output").iterdir())) == 5


class ReadFacade:
    def __init__(self, payload, tag):
        self.payload, self.tag = payload, tag

    def read_complete(self, _volume, path, *, max_bytes):
        return self.tag if path.endswith(".mac") else self.payload


class ReadVerifier:
    def verify(self, purpose, payload, tag, key_ref):
        return purpose == "modal-packaged-evaluation/v1" and key_ref == "key" and tag == hashlib.sha256(payload).digest()


def test_reader_requires_signed_record_bound_to_exact_workload_and_adapter(monkeypatch):
    baseline = _case()[2]
    document = json.loads(baseline)
    document["configuration"]["document"]["post_training"] = _evaluation()
    compiled = compile_packaged_sft_workload(resolved_config=CanonicalDocument.from_mapping(
        document["configuration"]["document"]))
    config = document["configuration"]["document"]["post_training"]
    binding = SimpleNamespace(
        execution_binding=SimpleNamespace(workload_digest=compiled.fingerprint,
                                          binding_digest="b" * 64),
        command_digest="c" * 64,
        provider_facts=SimpleNamespace(artifact_volume_id="vo-artifact"),
    )
    completion = SimpleNamespace(
        effect_id="effect", provider_job_ref="fc-1", completion_digest="d" * 64,
        members=(SimpleNamespace(role="final_model", sha256="e" * 64),),
    )
    record = _record(config, workload_digest=compiled.fingerprint,
                     adapter_digest="e" * 64)
    unit = "A distant bell answered. "
    large_response = unit * ((post_training_eval.MAX_EVALUATION_RESPONSE_BYTES - 512) // len(unit)) + "🔔"
    assert post_training_eval.MAX_EVALUATION_RESPONSE_BYTES - 1024 < len(large_response.encode("utf-8")) < post_training_eval.MAX_EVALUATION_RESPONSE_BYTES
    record["cases"][0]["response"] = large_response
    payload = post_training_eval.canonical_evaluation_document_bytes({
        "schema_version": "synaptic-modal-packaged-evaluation/v1",
        "effect_id": "effect", "command_digest": binding.command_digest,
        "provider_job_ref": "fc-1", "execution_binding_digest": "b" * 64,
        "training_completion_sha256": completion.completion_digest,
        "post_training_sha256": hashlib.sha256(canonical_bytes(config)).hexdigest(),
        "evaluation": record,
    })
    assert len(payload) > post_training_eval.MAX_EVALUATION_RESPONSE_BYTES
    reader = ModalPackagedReader.__new__(ModalPackagedReader)
    reader._facade = ReadFacade(payload, hashlib.sha256(payload).digest())
    reader._verifier = ReadVerifier()
    reader._key_ref = "key"
    monkeypatch.setattr(ModalPackagedReader, "observe_completion",
                        lambda self, _binding, *, provider_job_ref: completion)
    readback = reader.read_evaluation(binding, provider_job_ref="fc-1",
                                      workload_bytes=compiled.canonical_bytes)
    assert readback == payload
    assert json.loads(readback)["evaluation"]["cases"][0]["response"] == large_response
    reader._facade = ReadFacade(payload, b"wrong")
    with pytest.raises(ValueError, match="authentication"):
        reader.read_evaluation(binding, provider_job_ref="fc-1",
                               workload_bytes=compiled.canonical_bytes)
    reader._facade = ReadFacade(payload, hashlib.sha256(payload).digest())
    completion.members = (SimpleNamespace(role="final_model", sha256="0" * 64),)
    with pytest.raises(ValueError, match="model binding mismatch"):
        reader.read_evaluation(binding, provider_job_ref="fc-1",
                               workload_bytes=compiled.canonical_bytes)
    completion.members = (SimpleNamespace(role="final_model", sha256="e" * 64),)
    binding.execution_binding.workload_digest = "0" * 64
    with pytest.raises(ValueError, match="workload differs"):
        reader.read_evaluation(binding, provider_job_ref="fc-1",
                               workload_bytes=compiled.canonical_bytes)
