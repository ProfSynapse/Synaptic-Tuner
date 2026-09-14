"""Fractional workload settings survive the authenticated training/chat chain."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from tuner.training.contracts import CanonicalDocument
from tuner.execution.providers.modal.coordinator_worker import (
    _execute_modal_worker,
    validate_modal_worker_invocation,
)
from tuner.execution.providers.modal.worker_ports import ModalProcessResult
from tests.execution.providers import test_modal_coordinator_bundle as bundle_cases
from tests.execution.providers.test_modal_coordinator_worker import admit, real_case
from tests.execution.providers.test_modal_inference_workload import _case, _bind
from tests.execution.providers.test_modal_inference_preparation import _prepared


@pytest.fixture
def fractional_configuration(monkeypatch):
    original = bundle_cases._config

    def configuration(**kwargs):
        document = original(**kwargs).to_dict()
        document["sft"]["learning_rate"] = 0.0002
        document["lora"] = {"dropout": 0.05}
        return CanonicalDocument.from_mapping(document)

    monkeypatch.setattr(bundle_cases, "_config", configuration)


def assert_fractional_workload(raw):
    configuration = json.loads(raw)["configuration"]["document"]
    assert configuration["sft"]["learning_rate"] == 0.0002
    assert configuration["lora"]["dropout"] == 0.05


def test_worker_admission_and_revalidation_preserve_fractional_workload(
    monkeypatch, fractional_configuration
):
    invocation = admit(real_case(monkeypatch))
    validate_modal_worker_invocation(invocation)
    assert_fractional_workload(invocation.workload)
    assert (
        invocation.submit_command.preparation.workload_digest
        == hashlib.sha256(
            b"synaptic-training-workload/v1\0" + invocation.workload
        ).hexdigest()
    )
    worker = "tuner.execution.providers.modal.coordinator_worker.worker_source."
    monkeypatch.setattr(
        worker + "read_locked_closure_manifest", lambda _: invocation.closure_manifest
    )
    monkeypatch.setattr(worker + "write_runtime_closure_manifest", lambda *args: None)
    monkeypatch.setattr(worker + "stage_runtime_worker", lambda *args: None)
    events = []

    def run(argv, **kwargs):
        assert kwargs["stdin"] == invocation.workload
        events.append("process")
        return ModalProcessResult(0)

    result = _execute_modal_worker(
        invocation,
        sources=SimpleNamespace(
            prepare_and_verify=lambda *args: events.append("source")
        ),
        processes=SimpleNamespace(run=run),
        commit_prepared=lambda: None,
    )
    assert result == ModalProcessResult(0)
    assert events == ["source", "process"]


def test_native_workload_binding_preserves_fractional_settings_without_body_reads(
    monkeypatch, fractional_configuration
):
    source, transport, values = _case(monkeypatch)
    workload = _bind(source, transport)
    assert_fractional_workload(workload.workload_bytes)
    assert values[6] == []


def test_chat_preparation_preserves_fractional_training_settings(
    monkeypatch, fractional_configuration
):
    preparation, source, workload, _, _ = _prepared(monkeypatch)
    assert_fractional_workload(workload.workload_bytes)
    assert preparation.preparation.source_digest == source.artifact_source_digest
