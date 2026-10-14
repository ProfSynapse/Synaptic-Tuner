"""Actual format custody and named recipe selection; no provider execution."""
from copy import deepcopy
import shutil
import socket
from pathlib import Path

import pytest
import yaml

from tests.dataset_prep.test_conversation_messages import build
from tests.contract.test_grpo_training_input_v1 import document as grpo_document
from tuner.dataset_prep.conversation_messages import publish_conversation_dataset
from tuner.training.grpo_preparation import GRPONormalizerConfigV1, GRPOTrainingInputNormalizerV1
from tuner.training.modal_host_prepared_copy import (
    ModalPreparedCopyUnavailable, _verified_identity, inspect_mounted_prepared_dataset,
    modal_copied_preparation_ports, stage_published_modal_dataset,
)
from tuner.training.modal_recipe import plan_modal_sft_recipe
from tuner.training.input_preparation import PosixPrivatePreparedRootAuthorityV1, TrainingInputPreparationServiceV1
from synaptic_tuner.api.v1.training_input import TrainingInputV1
from synaptic_tuner.api.v1.training_sources import LocalTrainingInputPathV1, TrainingPreparationConfigV1

ROOT = Path(__file__).parents[2]
RECIPE = ROOT / "Trainers/recipes/qwen35_4b_32k_modal_prompt_completion.yaml"
PROFILES = ROOT / "Trainers/runtime_profiles"
CONVERSATION = "synaptic-conversation-sft-row/v1"


def conversation_recipe(tmp_path, monkeypatch, *, profile="qwen35-conversation-sft-candidate-v1"):
    root = tmp_path / ".tracking" / "datasets"
    root.mkdir(parents=True, mode=0o700)
    published = publish_conversation_dataset(root, build())
    document = yaml.safe_load(RECIPE.read_text())
    document["job"]["runtime_profile"] = profile
    document["dataset"].update(local_file=str(published.path.relative_to(tmp_path) / "dataset.jsonl"),
        schema_version=CONVERSATION, expected_digest=published.semantic_identity.dataset_digest,
        expected_split_counts=dict(published.semantic_identity.split_counts))
    document["training"]["require_memory_efficient_loss"] = True
    from tuner.training import modal_recipe
    monkeypatch.setattr(modal_recipe, "load_recipe", lambda *_: deepcopy(document))
    monkeypatch.setattr(socket.socket, "connect", lambda *_: pytest.fail("no provider/network access"))
    return published, document


def test_named_conversation_recipe_real_plan_and_private_custody(tmp_path, monkeypatch):
    published, document = conversation_recipe(tmp_path, monkeypatch)
    plan = plan_modal_sft_recipe(RECIPE, project_root=tmp_path, profiles_root=PROFILES)
    assert plan.prepared_identity.format == CONVERSATION
    assert plan.recipe.prepared_format == CONVERSATION
    assert plan.recipe.runtime_profile == document["job"]["runtime_profile"]
    assert plan.workload_digest and plan.configuration_digest
    config = plan.recipe.packaged_config(plan.prepared_identity).to_dict()
    assert config["dataset"] == plan.prepared_identity.to_dict()
    assert config["sft"]["require_memory_efficient_loss"] is True
    original = (published.path / "dataset.jsonl").read_bytes()
    # Model a nonprivate mounted publication; private admission must still reject it.
    published.path.chmod(0o755)
    for member in published.path.iterdir():
        member.chmod(0o644)
    from tuner.training.conversation_preparation import ConversationPreparedInputFormatVerifierV1
    with pytest.raises(ValueError):
        ConversationPreparedInputFormatVerifierV1().verify(published.path)
    private = tmp_path / "private-run"
    copied = stage_published_modal_dataset(published.path, private_root=private,
        expected_dataset_digest=plan.prepared_identity.revision,
        expected_content_digest=plan.prepared_identity.content_digest, prepared_format=CONVERSATION)
    normalizer, normalizer_config, verifier = modal_copied_preparation_ports(
        private_root=private, identity=plan.prepared_identity,
        train_rows=plan.recipe.train_rows, validation_rows=plan.recipe.validation_rows)
    service = TrainingInputPreparationServiceV1(prepared_root=private,
        normalizers={normalizer_config.normalizer_ref: normalizer},
        format_verifiers={verifier.format: verifier}, root_authority=PosixPrivatePreparedRootAuthorityV1())
    result = service.prepare(LocalTrainingInputPathV1(copied), TrainingPreparationConfigV1(
        "conversation-request", "standalone-training",
        plan.recipe.training_input(plan.prepared_identity.ref).canonical_json(), normalizer_config))
    assert result.prepared.identity == plan.prepared_identity
    shutil.rmtree(published.path)
    lease = result.retained_source.open_lease()
    try:
        assert lease.take_stream().read() == original
    finally:
        lease.close()


def test_conversation_cannot_select_v2_only_build(tmp_path, monkeypatch):
    conversation_recipe(tmp_path, monkeypatch, profile="qwen35-sft-v1")
    with pytest.raises(ValueError, match="capability"):
        plan_modal_sft_recipe(RECIPE, project_root=tmp_path, profiles_root=PROFILES)


def test_wrong_explicit_format_and_memory_guard_reject(tmp_path, monkeypatch):
    _, document = conversation_recipe(tmp_path, monkeypatch)
    from tuner.training import modal_recipe
    document["dataset"]["schema_version"] = "unknown/v1"
    monkeypatch.setattr(modal_recipe, "load_recipe", lambda *_: document)
    with pytest.raises(ValueError, match="explicitly supported"):
        plan_modal_sft_recipe(RECIPE, project_root=tmp_path, profiles_root=PROFILES)
    document["dataset"]["schema_version"] = CONVERSATION
    document["training"]["require_memory_efficient_loss"] = False
    with pytest.raises(ValueError, match="memory-efficient"):
        plan_modal_sft_recipe(RECIPE, project_root=tmp_path, profiles_root=PROFILES)


def test_grpo_existing_format_ports_validate_and_preserve_bytes(tmp_path):
    source, output = tmp_path / "source", tmp_path / "published"
    source.mkdir(mode=0o700); output.mkdir(mode=0o700)
    raw = b'{"prompt":"Say OK","expected":"OK","fraction":0.25}\n'
    (source / "rows.jsonl").write_bytes(raw); (source / "rows.jsonl").chmod(0o600)
    publication = GRPOTrainingInputNormalizerV1(allowed_source_roots=(source,)).prepare(
        LocalTrainingInputPathV1(source), config=GRPONormalizerConfigV1("rows.jsonl", "prompt"), output_root=output)
    inspected = inspect_mounted_prepared_dataset(publication.path, prepared_format=publication.format)
    identity = _verified_identity(inspected, publication.format)
    publication.path.chmod(0o755)
    for member in publication.path.iterdir():
        member.chmod(0o644)
    from tuner.training.grpo_preparation import GRPOPreparedInputFormatVerifierV1
    with pytest.raises(ValueError):
        GRPOPreparedInputFormatVerifierV1().verify(publication.path)
    private = tmp_path / "private-grpo"
    copied = stage_published_modal_dataset(publication.path, private_root=private,
        expected_dataset_digest=identity.revision, expected_content_digest=identity.content_digest,
        prepared_format=identity.format)
    normalizer, config, verifier = modal_copied_preparation_ports(private_root=private, identity=identity)
    service = TrainingInputPreparationServiceV1(prepared_root=private,
        normalizers={config.normalizer_ref: normalizer}, format_verifiers={verifier.format: verifier},
        root_authority=PosixPrivatePreparedRootAuthorityV1())
    request = grpo_document(); request["dataset"] = {"ref": identity.ref}
    result = service.prepare(LocalTrainingInputPathV1(copied), TrainingPreparationConfigV1(
        "grpo-request", "host", TrainingInputV1.from_dict(request).canonical_json(), config))
    shutil.rmtree(publication.path)
    lease = result.retained_source.open_lease()
    try:
        assert lease.take_stream().read() == raw
    finally:
        lease.close()


def test_explicit_wrong_copy_format_has_no_copy(tmp_path, monkeypatch):
    published, _ = conversation_recipe(tmp_path, monkeypatch)
    private = tmp_path / "wrong-copy"
    with pytest.raises(ModalPreparedCopyUnavailable):
        stage_published_modal_dataset(published.path, private_root=private,
            expected_dataset_digest=published.semantic_identity.dataset_digest,
            prepared_format="syntunia-sft-row/v2")
    assert not private.exists() or list(private.iterdir()) == []


def test_copy_rejects_source_directory_substitution_without_adoption(tmp_path, monkeypatch):
    published, _ = conversation_recipe(tmp_path, monkeypatch)
    from tuner.training import modal_host_prepared_copy as copy
    original = copy._read_member
    replaced = False
    def read(directory, name, maximum):
        nonlocal replaced
        raw = original(directory, name, maximum)
        if name == "dataset.jsonl" and not replaced:
            replaced = True
            published.path.rename(published.path.with_name("retained-original"))
            published.path.mkdir(mode=0o700)
        return raw
    monkeypatch.setattr(copy, "_read_member", read)
    private = tmp_path / "substituted-copy"
    with pytest.raises(ModalPreparedCopyUnavailable):
        stage_published_modal_dataset(published.path, private_root=private,
            expected_dataset_digest=published.semantic_identity.dataset_digest, prepared_format=CONVERSATION)
    assert not private.exists() or list(private.iterdir()) == []


def test_source_inventory_extra_member_and_bad_hash_fail_closed(tmp_path, monkeypatch):
    published, _ = conversation_recipe(tmp_path, monkeypatch)
    (published.path / "unexpected").write_text("not an admitted member")
    with pytest.raises(ModalPreparedCopyUnavailable):
        inspect_mounted_prepared_dataset(published.path, prepared_format=CONVERSATION)
    (published.path / "unexpected").unlink()
    leaf = published.path / "dataset.jsonl"
    leaf.chmod(0o600); leaf.write_bytes(b'{"changed":true}\n')
    with pytest.raises(ModalPreparedCopyUnavailable):
        inspect_mounted_prepared_dataset(published.path, prepared_format=CONVERSATION)


def test_normal_runner_conversation_reaches_real_public_prepare_resolve_start(tmp_path, monkeypatch):
    import json
    from dataclasses import replace
    import tests.training.test_modal_standalone_runner as fixture
    import tuner.training.modal_standalone_runner as runner
    from tuner.execution.providers.modal.runtime_build import ModalBuildCandidateV1
    from synaptic_tuner.api.v1.training_facade import TrainingAPI
    # Extend only the existing scripted capture's declared format compatibility.
    # This remains fake provider/runtime state, not a qualified image observation.
    candidate = fixture._candidate
    def admitted_candidate():
        original = candidate()
        measured = original.measured
        measured["capabilities"]["compatibility"]["dataset_formats"] = [CONVERSATION, "syntunia-sft-row/v2"]
        raw = (json.dumps({"schema_version": "synaptic-modal-build-capture/v1",
            "image_id": original.image_id, "inspector_sha256": original.inspector_sha256,
            "measured": measured}, sort_keys=True, separators=(",", ":")) + "\n").encode()
        return ModalBuildCandidateV1.from_capture(material=original.material, raw=raw,
            expected_image_id=original.image_id, expected_inspector_sha256=original.inspector_sha256,
            expected_build_inputs_digest=measured["build_inputs_digest"])
    monkeypatch.setattr(fixture, "_candidate", admitted_candidate)
    _, context, events = fixture._setup(tmp_path, monkeypatch)
    root = tmp_path / ".tracking" / "datasets"
    root.chmod(0o700)
    published = publish_conversation_dataset(root, build())
    document = yaml.safe_load(RECIPE.read_text())
    document["job"]["runtime_profile"] = "qwen35-conversation-sft-candidate-v1"
    document["dataset"].update(local_file=str(published.path.relative_to(tmp_path) / "dataset.jsonl"),
        schema_version=CONVERSATION, expected_digest=published.semantic_identity.dataset_digest,
        expected_split_counts=dict(published.semantic_identity.split_counts))
    document["training"]["require_memory_efficient_loss"] = True
    for name in ("qwen35-conversation-sft-candidate-v1.yaml", "qwen35-conversation-sft-candidate-v1.compatibility.json"):
        shutil.copy2(PROFILES / name, tmp_path / "Trainers/runtime_profiles" / name)
    shutil.copytree(ROOT / "Trainers/image_profiles/qwen35_4b_conversation_sft_candidate_v1",
                    tmp_path / "Trainers/image_profiles/qwen35_4b_conversation_sft_candidate_v1")
    from tuner.training import modal_recipe
    monkeypatch.setattr(modal_recipe, "load_recipe", lambda *_: document)
    plan = plan_modal_sft_recipe(RECIPE, project_root=tmp_path,
                                 profiles_root=tmp_path / "Trainers/runtime_profiles")
    reached = []
    original_prepare, original_resolve, original_start = TrainingAPI.prepare, TrainingAPI.resolve, TrainingAPI.start
    def prepare(api, source, config):
        result = original_prepare(api, source, config)
        assert result.prepared.identity == plan.prepared_identity
        reached.append("prepare")
        return result
    def resolve(api, request):
        assert TrainingInputV1.from_json(request.canonical_json).dataset.ref == plan.prepared_identity.ref
        result = original_resolve(api, request)
        assert result.workload_digest == plan.workload_digest
        reached.append("resolve")
        return result
    def start(api, public_plan, preflight):
        assert public_plan.basis.workload_digest == plan.workload_digest
        reached.append("start")
        return original_start(api, public_plan, preflight)
    monkeypatch.setattr(TrainingAPI, "prepare", prepare)
    monkeypatch.setattr(TrainingAPI, "resolve", resolve)
    monkeypatch.setattr(TrainingAPI, "start", start)
    result = runner.run_modal_standalone_job(plan=plan, context=context,
        modal_profile="explicit", modal_environment="main")
    assert reached == ["prepare", "resolve", "start"]
    assert events.count("stage") == 1 and events.count("spawn") == 1
    assert result.run.project_ref == "standalone-training"
    # A v2-only named build cannot be used with this conversation input.
    events.clear()
    wrong_recipe = replace(plan.recipe, runtime_profile="qwen35-sft-v1")
    wrong_plan = replace(plan, recipe=wrong_recipe)
    with pytest.raises(ValueError, match="capability"):
        runner._qualified_runtime(modal=object(), plan=wrong_plan, context=context,
            resources=object(), storage=object(), effect_id="unused", modal_profile="explicit",
            modal_environment="main", builder_cache_root=tmp_path)
    assert events == []
