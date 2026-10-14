import json
import os
from dataclasses import replace

import pytest

from synaptic_tuner.api.v1.training_input import TrainingInputV1
from synaptic_tuner.api.v1.training_sources import LocalTrainingInputPathV1, TrainingPreparationConfigV1
from tuner.training.grpo_preparation import (
    GRPONormalizerConfigV1, GRPOPreparedInputFormatVerifierV1, GRPOTrainingInputNormalizerV1,
    snapshot_grpo_publication,
)
from tuner.training.input_preparation import PosixPrivatePreparedRootAuthorityV1, TrainingInputPreparationServiceV1
from tests.contract.test_grpo_training_input_v1 import document


def setup(tmp_path, raw=b'{"question":"Say OK","expected":"OK","fraction":0.5}\n'):
    source, output = tmp_path / "source", tmp_path / "prepared"
    source.mkdir(mode=0o700); output.mkdir(mode=0o700)
    path = source / "rows.jsonl"; path.write_bytes(raw); path.chmod(0o600)
    normalizer = GRPOTrainingInputNormalizerV1(allowed_source_roots=(source,))
    verifier = GRPOPreparedInputFormatVerifierV1()
    service = TrainingInputPreparationServiceV1(prepared_root=output,
        normalizers={"synaptic.grpo-prompts/v1": normalizer},
        format_verifiers={verifier.format: verifier}, root_authority=PosixPrivatePreparedRootAuthorityV1())
    doc = document(); doc["hyperparameters"]["prompt_column"] = "question"
    config = TrainingPreparationConfigV1("prepare-1", "project", TrainingInputV1.from_dict(doc).canonical_json(),
        GRPONormalizerConfigV1("rows.jsonl", "question"))
    return service, LocalTrainingInputPathV1(source), config, raw, output, verifier


def test_real_service_preserves_original_and_reuses_across_models(tmp_path):
    service, source, config, raw, output, verifier = setup(tmp_path)
    result = service.prepare(source, config)
    identity = result.prepared.identity
    path = output / ("dataset-" + identity.revision)
    manifest, retained = snapshot_grpo_publication(path)
    assert retained == raw
    assert manifest["prompt_column"] == "question"
    assert not set(manifest) & {"split_counts", "row_ids_sha256", "model", "rewards"}
    second_doc = json.loads(config.canonical_training_json)
    second_doc["model"]["ref"] = "model/another"
    second_doc["hyperparameters"]["learning_rate"] = 1e-5
    second_doc["rewards"]["documents"]["exact"]["validation"][0]["match"] = "^YES$"
    second = service.prepare(source, replace(config, request_id="prepare-2",
        canonical_training_json=TrainingInputV1.from_dict(second_doc).canonical_json()))
    assert second.prepared.identity == identity
    lease = second.retained_source.open_lease()
    try:
        assert lease.take_stream().read() == raw
    finally:
        lease.close()


def test_request_prompt_mismatch_rejected_before_source_lease(tmp_path, monkeypatch):
    service, source, config, _, _, verifier = setup(tmp_path)
    doc = json.loads(config.canonical_training_json); doc["hyperparameters"]["prompt_column"] = "other"
    monkeypatch.setattr(verifier, "source", lambda *args: pytest.fail("lease must not open"))
    with pytest.raises(ValueError, match="prompt column"):
        service.prepare(source, replace(config, canonical_training_json=TrainingInputV1.from_dict(doc).canonical_json()))


@pytest.mark.parametrize("mutation", ["bytes", "permission", "symlink", "ancestor"])
def test_publication_mutation_blocks_new_lease(tmp_path, mutation):
    service, source, config, _, output, _ = setup(tmp_path)
    result = service.prepare(source, config)
    path = output / ("dataset-" + result.prepared.identity.revision)
    leaf = path / "dataset.jsonl"
    if mutation == "bytes":
        leaf.chmod(0o600); leaf.write_bytes(b'{"question":"changed"}\n')
    elif mutation == "permission":
        leaf.chmod(0o644)
    elif mutation == "symlink":
        leaf.unlink(); leaf.symlink_to(tmp_path / "missing")
    else:
        path.rename(output / "original"); path.mkdir(mode=0o700)
    with pytest.raises((ValueError, OSError)):
        result.retained_source.open_lease()


def test_unverified_existing_destination_is_not_replaced(tmp_path):
    service, source, config, _, output, _ = setup(tmp_path)
    result = service.prepare(source, config)
    path = output / ("dataset-" + result.prepared.identity.revision)
    (path / "unexpected").write_text("collision")
    with pytest.raises(ValueError):
        service.prepare(source, config)
    assert (path / "unexpected").read_text() == "collision"


def test_source_root_substitution_rejects(tmp_path):
    service, source, config, _, _, _ = setup(tmp_path)
    root = tmp_path / "source"
    root.rename(tmp_path / "old-source"); root.mkdir(mode=0o700)
    (root / "rows.jsonl").write_bytes(b'{"question":"Say OK"}\n')
    os.chmod(root / "rows.jsonl", 0o600)
    with pytest.raises(ValueError):
        service.prepare(source, config)


def test_optional_verifier_runs_core_admission_before_lease(tmp_path, monkeypatch):
    import tuner.training.grpo_preparation as preparation
    service, source, config, _, _, verifier = setup(tmp_path)
    calls = []
    original = preparation.admit_grpo_snapshot
    def checked(value, identity, raw):
        calls.append((value, identity, raw))
        return original(value, identity, raw)
    monkeypatch.setattr(preparation, "admit_grpo_snapshot", checked)
    result = service.prepare(source, config)
    assert len(calls) == 1
    assert calls[0][1] == result.prepared.identity
    assert calls[0][0].dataset.ref == result.prepared.identity.ref


def test_partial_publication_failure_retains_owned_stage(tmp_path, monkeypatch):
    from tuner.dataset_prep import publication as storage
    service, source, config, _, output, _ = setup(tmp_path)
    original = storage._write_stage_member
    def failing(stage, name, raw):
        if name == "manifest.json":
            raise OSError("controlled failure")
        return original(stage, name, raw)
    monkeypatch.setattr(storage, "_write_stage_member", failing)
    with pytest.raises(OSError):
        service.prepare(source, config)
    stages = list(output.iterdir())
    assert len(stages) == 1 and stages[0].name.startswith(".dataset-")
    assert (stages[0] / "dataset.jsonl").is_file()
    assert not (stages[0] / "manifest.json").exists()


def test_coordination_constructor_failure_closes_output_authority(tmp_path, monkeypatch):
    from tuner.dataset_prep import publication as storage
    service, source, config, _, output, _ = setup(tmp_path)
    seen = []
    def rejected(root):
        seen.append(root)
        raise OSError("controlled coordination failure")
    monkeypatch.setattr(storage, "_OutputRootCoordination", rejected)
    with pytest.raises(OSError):
        service.prepare(source, config)
    assert len(seen) == 1 and seen[0].path == output and seen[0].closed
    assert list(output.iterdir()) == []


def test_close_failure_is_not_success(tmp_path, monkeypatch):
    from tuner.dataset_prep import publication as storage
    service, source, config, _, output, _ = setup(tmp_path)
    original = storage._OutputRootAuthority.close
    def closing(root):
        result = original(root)
        return False if root.path == output else result
    monkeypatch.setattr(storage._OutputRootAuthority, "close", closing)
    with pytest.raises(ValueError, match="authority close"):
        service.prepare(source, config)
    assert len(list(output.iterdir())) == 1  # committed publication remains; no cleanup authority inferred


def test_source_member_fifo_rejects_without_blocking(tmp_path):
    service, source, config, _, _, _ = setup(tmp_path)
    leaf = tmp_path / "source" / "rows.jsonl"
    leaf.unlink(); os.mkfifo(leaf, 0o600)
    with pytest.raises(ValueError, match="regular file"):
        service.prepare(source, config)
