from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from synaptic_tuner.api.v1 import (
    RunArtifactRequest,
    RunListRequest,
    RunOperationCode,
    RunOperationError,
    RunsAPI,
    TrainingRunRef,
    TrainingRunState,
)
from synaptic_tuner.api.v1.execution import ExecutionGrant
from synaptic_tuner.api.v1.modal import compose_modal_verified_run_reads
from tests.execution.providers.test_modal_sdk154_adapter import FakeVolume
from tests.execution.providers.test_modal_training_operations import (
    _publish_unrelated_completed_run,
    operations,
    profile,
)
from tuner.execution.providers.modal.contracts import ArtifactRole
from tuner.runtime.verification import VerificationStatus as RuntimeVerificationStatus


RUN = TrainingRunRef("run-1", "project-1")


def _verified_reads(tmp_path):
    training, repository, plan = operations(tmp_path)
    submission = training.start(
        plan, training.preflight(plan), ExecutionGrant("grant-run-1")
    )
    _publish_unrelated_completed_run(training, repository, tmp_path)
    training._verify_semantics = (
        lambda _preparation, _manifest: RuntimeVerificationStatus.VERIFIED
    )
    assert training.outcome(submission).status.state.value == "succeeded"
    reads = compose_modal_verified_run_reads(
        context=training._context,
        repository=training._repository,
        authenticator=training._ports.authenticator,
        modal_reads=training._facade,
        clock=training._ports.clock,
    )
    return RunsAPI(reads), reads, repository


def test_projects_exact_verified_inventory_without_mutating_lifecycle(tmp_path):
    api, _reads, repository = _verified_reads(tmp_path)
    before = repository.load("project-1", "run-1").canonical_bytes

    outcome = api.show(RUN)
    verification = api.reverify(RUN)

    assert outcome.state is TrainingRunState.SUCCEEDED
    assert tuple(item.role for item in outcome.artifacts) == tuple(
        sorted(role.value for role in ArtifactRole)
    )
    assert verification.verified is True
    assert repository.load("project-1", "run-1").canonical_bytes == before


def test_stream_is_single_use_and_rechecks_digest_and_size(tmp_path):
    api, _reads, _repository = _verified_reads(tmp_path)
    expected = b"unrelated-final_model"
    stream = api.artifacts(RunArtifactRequest(RUN, "final_model", len(expected)))

    assert b"".join(stream.iter_bytes()) == expected
    with pytest.raises(RunOperationError) as reused:
        list(stream.iter_bytes())
    assert reused.value.code is RunOperationCode.ARTIFACT_CONTENT_INVALID


def test_same_size_mutation_after_manifest_snapshot_is_rejected(tmp_path):
    api, _reads, _repository = _verified_reads(tmp_path)
    expected = b"unrelated-final_model"
    stream = api.artifacts(RunArtifactRequest(RUN, "final_model", len(expected)))
    path = "operations/effect-run-1/output/final_model"
    FakeVolume.registry["artifact-name"].files[path] = b"x" * len(expected)

    with pytest.raises(RunOperationError) as changed:
        list(stream.iter_bytes())
    assert changed.value.code is RunOperationCode.ARTIFACT_CONTENT_INVALID


def test_stream_rejects_provider_chunk_above_public_bound(tmp_path, monkeypatch):
    api, reads, _repository = _verified_reads(tmp_path)
    stream = api.artifacts(RunArtifactRequest(RUN, "final_model", 2_000_000))

    def oversized(_self, _volume_id, _path, *, max_bytes):
        assert max_bytes == 2_000_000
        yield b"x" * 1_048_577

    monkeypatch.setattr(type(reads._facade), "iter_complete", oversized)
    with pytest.raises(RunOperationError) as invalid:
        list(stream.iter_bytes())
    assert invalid.value.code is RunOperationCode.ARTIFACT_CONTENT_INVALID


@pytest.mark.parametrize("mode", ["empty", "short", "excess", "failure"])
def test_stream_closes_malformed_provider_results(tmp_path, monkeypatch, mode):
    api, reads, _repository = _verified_reads(tmp_path)
    expected = b"unrelated-final_model"
    stream = api.artifacts(RunArtifactRequest(RUN, "final_model", 1000))

    def malformed(_self, _volume_id, _path, *, max_bytes):
        if mode == "failure":
            raise RuntimeError("raw provider failure")
        if mode == "empty":
            return
        if mode == "short":
            yield expected[:-1]
        else:
            yield expected + b"x"

    monkeypatch.setattr(type(reads._facade), "iter_complete", malformed)
    with pytest.raises(RunOperationError) as invalid:
        list(stream.iter_bytes())
    assert invalid.value.code is RunOperationCode.ARTIFACT_CONTENT_INVALID


def test_stream_public_bindings_are_frozen(tmp_path):
    api, _reads, _repository = _verified_reads(tmp_path)
    stream = api.artifacts(RunArtifactRequest(RUN, "final_model", 1000))
    for name, value in (
        ("run", TrainingRunRef("other", "project-1")),
        ("artifact", stream.artifact),
        ("maximum_bytes", 1),
        ("_path", "other"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(stream, name, value)
    assert not hasattr(stream, "_state")


def test_concurrent_stream_claim_has_exactly_one_winner(tmp_path):
    api, _reads, _repository = _verified_reads(tmp_path)
    stream = api.artifacts(RunArtifactRequest(RUN, "final_model", 1000))
    barrier = Barrier(2)

    def claim():
        barrier.wait()
        try:
            iterator = stream.iter_bytes()
        except RunOperationError as error:
            return error.code
        return b"".join(iterator)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _index: claim(), range(2)))
    assert results.count(b"unrelated-final_model") == 1
    assert results.count(RunOperationCode.ARTIFACT_CONTENT_INVALID) == 1


def test_verified_inventory_is_reused_until_explicit_reverify(tmp_path, monkeypatch):
    api, reads, _repository = _verified_reads(tmp_path)
    plane_type = type(reads._completion)
    original = plane_type.validate
    calls = []

    def counted(self, effect_id):
        calls.append(effect_id)
        return original(self, effect_id)

    monkeypatch.setattr(plane_type, "validate", counted)
    api.show(RUN)
    api.show(RUN)
    streams = [
        api.artifacts(RunArtifactRequest(RUN, role.value, 1000))
        for role in ArtifactRole
    ]
    assert len(streams) == 5
    assert calls == ["effect-run-1"]
    api.reverify(RUN)
    assert calls == ["effect-run-1", "effect-run-1"]


def test_ledger_drift_after_stream_creation_refuses_before_provider_io(tmp_path):
    api, _reads, repository = _verified_reads(tmp_path)
    stream = api.artifacts(RunArtifactRequest(RUN, "final_model", 1000))
    record = repository.load("project-1", "run-1")
    repository.records[("project-1", "run-1")] = replace(
        record, updated_at="2026-08-25T12:03:01Z"
    )
    calls = list(FakeVolume.calls)
    with pytest.raises(RunOperationError) as drifted:
        list(stream.iter_bytes())
    assert drifted.value.code is RunOperationCode.ARTIFACT_CONTENT_INVALID
    assert FakeVolume.calls == calls


def test_cross_project_run_refuses_before_provider_io(tmp_path):
    api, _reads, _repository = _verified_reads(tmp_path)
    calls = list(FakeVolume.calls)
    with pytest.raises(RunOperationError) as refused:
        api.show(TrainingRunRef("run-1", "other-project"))
    assert refused.value.code is RunOperationCode.READ_INELIGIBLE
    assert FakeVolume.calls == calls


def test_effect_drift_refuses_before_provider_io(tmp_path):
    api, _reads, repository = _verified_reads(tmp_path)
    record = repository.load("project-1", "run-1")
    effect = record.effects[0]
    changed_effect = replace(
        effect, identity=replace(effect.identity, effect_id="effect-other")
    )
    repository.records[("project-1", "run-1")] = replace(
        record, effects=(changed_effect,)
    )
    calls = list(FakeVolume.calls)
    with pytest.raises(RunOperationError) as refused:
        api.show(RUN)
    assert refused.value.code is RunOperationCode.READ_INELIGIBLE
    assert FakeVolume.calls == calls


def test_authenticated_metadata_drift_is_rejected_on_refresh(tmp_path):
    api, _reads, _repository = _verified_reads(tmp_path)
    api.show(RUN)
    path = "operations/effect-run-1/output/final_model"
    FakeVolume.registry["artifact-name"].files[path] += b"x"
    with pytest.raises(RunOperationError) as drifted:
        api.reverify(RUN)
    assert drifted.value.code is RunOperationCode.PROVIDER_READ_INVALID


def test_failed_reverify_evicts_prior_verified_inventory(tmp_path, monkeypatch):
    api, reads, _repository = _verified_reads(tmp_path)
    api.show(RUN)
    plane_type = type(reads._completion)
    original = plane_type.validate
    calls = []

    def counted(self, effect_id):
        calls.append(effect_id)
        return original(self, effect_id)

    monkeypatch.setattr(plane_type, "validate", counted)
    path = "operations/effect-run-1/output/final_model"
    FakeVolume.registry["artifact-name"].files[path] += b"x"
    with pytest.raises(RunOperationError):
        api.reverify(RUN)
    with pytest.raises(RunOperationError):
        api.artifacts(RunArtifactRequest(RUN, "tokenizer", 1000))
    assert calls == ["effect-run-1", "effect-run-1"]


def test_role_and_caller_limit_are_closed(tmp_path):
    api, _reads, _repository = _verified_reads(tmp_path)
    with pytest.raises(RunOperationError) as missing:
        api.artifacts(RunArtifactRequest(RUN, "missing", 1000))
    assert missing.value.code is RunOperationCode.ARTIFACT_ROLE_MISSING
    with pytest.raises(RunOperationError) as bounded:
        api.artifacts(RunArtifactRequest(RUN, "final_model", 1))
    assert bounded.value.code is RunOperationCode.ARTIFACT_LIMIT_EXCEEDED


def test_non_verified_run_refuses_before_completion_io(tmp_path):
    api, reads, repository = _verified_reads(tmp_path)
    record = repository.load("project-1", "run-1")
    repository.records[("project-1", "run-1")] = replace(
        record, verification=record.verification.NOT_READY
    )
    calls = list(FakeVolume.calls)

    with pytest.raises(RunOperationError) as refused:
        api.show(RUN)
    assert refused.value.code is RunOperationCode.READ_INELIGIBLE
    assert FakeVolume.calls == calls


def test_unsupported_operations_are_explicitly_unavailable(tmp_path):
    _api, reads, _repository = _verified_reads(tmp_path)
    for invoke in (
        lambda: reads.list(RunListRequest("project-1")),
        lambda: reads.logs(object()),
        lambda: reads.cancel(RUN, "reason"),
        lambda: reads.reconcile(RUN),
    ):
        with pytest.raises(RunOperationError) as unavailable:
            invoke()
        assert unavailable.value.code is RunOperationCode.CAPABILITY_UNAVAILABLE
