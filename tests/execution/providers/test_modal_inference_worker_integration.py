"""Mounted chat preparation reuses the actual Modal pinned-model adapter."""

import pytest

from tests.execution.providers.modal_inference_worker_fixtures import (
    mounted_launch_case,
)
from tuner.execution.providers.modal import inference_model, inference_worker
from tuner.execution.providers.modal.inference_model import ModalPinnedModelPreparer
from tuner.execution.providers.modal.inference_worker import prepare_modal_chat_worker
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
    target = prepare_modal_chat_worker(
        case.envelope.argument_bytes, **(case.kwargs | {"preparer": preparer})
    )
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
