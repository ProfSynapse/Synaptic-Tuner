from __future__ import annotations

from copy import deepcopy
from argparse import Namespace
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest
import yaml

from synaptic_tuner.api.v1._contract import PreparedTrainingInputIdentity
from synaptic_tuner.api.v1.training_sources import (
    LocalTrainingInputPathV1, TrainingPreparationConfigV1,
)
from tests.contract.test_public_training_input_v1 import _document
from tests.dataset_prep.test_context_messages_v2 import _config
from tests.training.test_input_preparation import _ExplicitTestRootAuthority
from tuner.dataset_prep import DatasetPublicationUncertainV1, prepare_dataset_v2
from tuner.training.input_preparation import (
    PUBLISHED_PREPARED_DATASET_NORMALIZER_V1,
    PublishedPreparedDatasetNormalizerConfigV1,
    PublishedPreparedDatasetNormalizerV1,
    TrainingInputPreparationServiceV1, default_dataset_format_verifiers_v1,
)
from tuner.training.modal_recipe import load_modal_sft_recipe, plan_modal_sft_recipe
from tuner.handlers.train_handler import TrainHandler
from tuner.training.packaged_compilation import compile_packaged_sft_workload


ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "Trainers/recipes/qwen35_4b_32k_modal_prompt_completion.yaml"
PROFILES = ROOT / "Trainers/runtime_profiles"


def _load(monkeypatch, mutation):
    from tuner.training import modal_recipe

    document = deepcopy(yaml.safe_load(RECIPE.read_text(encoding="utf-8")))
    mutation(document)
    monkeypatch.setattr(modal_recipe, "load_recipe", lambda _path, _runner: document)
    return load_modal_sft_recipe(RECIPE, profiles_root=PROFILES)


def test_recipe_maps_to_exact_public_and_packaged_sft_controls():
    recipe = load_modal_sft_recipe(RECIPE, profiles_root=PROFILES)
    identity = PreparedTrainingInputIdentity(
        "prepared://sha256/" + recipe.dataset_digest,
        recipe.dataset_digest, "a" * 64, 123,
        "syntunia-sft-row/v2",
    )
    public = recipe.training_input(identity.ref)
    config = recipe.packaged_config(identity)
    workload = compile_packaged_sft_workload(resolved_config=config)
    assert config.to_dict()["sft"] == public.hyperparameters.to_dict()
    assert config.to_dict()["dataset"] == identity.to_dict()
    assert public.model.revision == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert public.hyperparameters.max_seq_length == 32768
    assert public.hyperparameters.lora_rank == 32
    assert public.hyperparameters.duration.max_steps == 2
    assert (recipe.accelerator, recipe.accelerator_count, recipe.timeout_seconds) == (
        "A100-80GB", 1, 1800,
    )
    assert recipe.maximum_cost_minor_units == 200
    assert workload.fingerprint


@pytest.mark.parametrize("mutation", [
    lambda data: data["job"].update(image="arbitrary/image"),
    lambda data: data["job"].update(runtime_profile="unknown"),
    lambda data: data["job"].update(accelerator_count=2),
    lambda data: data["job"].update(timeout_seconds=0),
    lambda data: data["job"].update(maximum_cost_minor_units=0),
    lambda data: data["job"].update(maximum_cost_minor_units="100"),
    lambda data: data["run"].update(dry_run=True),
    lambda data: data["run"].update(command=["python", "train.py"]),
    lambda data: data["model"].update(revision="0" * 40),
    lambda data: data["training"].update(api_token="secret"),
    lambda data: data["dataset"].update(local_file="../outside/dataset.jsonl"),
    lambda data: data["artifacts"].update(required_kinds=["final_model"]),
])
def test_recipe_rejects_unapproved_authority_and_incompatible_controls(monkeypatch, mutation):
    with pytest.raises((TypeError, ValueError)):
        _load(monkeypatch, mutation)


def test_existing_v2_publication_is_prepared_without_changing_identity(tmp_path):
    _, _, _, dataset_config = _config(tmp_path / "source")
    root = (tmp_path / "private-prepared").resolve()
    try:
        published = prepare_dataset_v2(dataset_config, root)
    except DatasetPublicationUncertainV1:
        published = prepare_dataset_v2(dataset_config, root)
    semantic = published.semantic_identity
    normalizer_config = PublishedPreparedDatasetNormalizerConfigV1(
        semantic.dataset_digest, semantic.split_counts["train"],
        semantic.split_counts["validation"],
    )
    service = TrainingInputPreparationServiceV1(
        prepared_root=root,
        normalizers={
            PUBLISHED_PREPARED_DATASET_NORMALIZER_V1:
                PublishedPreparedDatasetNormalizerV1(authorized_root=root),
        },
        format_verifiers=default_dataset_format_verifiers_v1(),
        root_authority=_ExplicitTestRootAuthority(),
    )
    from synaptic_tuner.api.v1.training_input import TrainingInputV1

    config = TrainingPreparationConfigV1(
        "request-a", "project-a",
        TrainingInputV1.from_dict(_document()).canonical_json(), normalizer_config,
    )
    result = service.prepare(LocalTrainingInputPathV1(published.path / "dataset.jsonl"), config)
    assert result.prepared.identity.revision == semantic.dataset_digest
    assert result.prepared.identity.content_digest == semantic.dataset_sha256
    with pytest.raises(ValueError, match="split counts"):
        service.prepare(
            LocalTrainingInputPathV1(published.path / "dataset.jsonl"),
            TrainingPreparationConfigV1(
                "request-a", "project-a", config.canonical_training_json,
                PublishedPreparedDatasetNormalizerConfigV1(
                    semantic.dataset_digest, normalizer_config.train_rows + 1,
                    normalizer_config.validation_rows,
                ),
            ),
        )


@pytest.mark.parametrize("missing", [True, False])
def test_train_plan_does_not_expose_private_recipe_paths_or_malformed_yaml(
    tmp_path, capsys, missing,
):
    private = tmp_path / "private-marker-secret-recipe.yaml"
    if not missing:
        private.write_text("secret-marker: [unterminated", encoding="utf-8")
    handler = TrainHandler(args=Namespace(job_config=str(private), plan=True, json=True))
    assert handler.handle() == 2
    output = capsys.readouterr().out
    assert "MODAL_RECIPE_INVALID" in output
    assert str(private) not in output
    assert "secret-marker" not in output


def test_plan_verifies_publication_without_network_or_provider_effects(tmp_path, monkeypatch):
    _, _, _, dataset_config = _config(tmp_path / "source")
    root = tmp_path / ".tracking" / "datasets"
    try:
        published = prepare_dataset_v2(dataset_config, root)
    except DatasetPublicationUncertainV1:
        published = prepare_dataset_v2(dataset_config, root)
    semantic = published.semantic_identity
    document = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
    document["dataset"]["local_file"] = str(
        Path(".tracking") / "datasets" / published.path.name / "dataset.jsonl"
    )
    document["dataset"]["expected_digest"] = semantic.dataset_digest
    document["dataset"]["expected_split_counts"] = dict(semantic.split_counts)
    from tuner.training import modal_recipe

    monkeypatch.setattr(modal_recipe, "load_recipe", lambda _path, _runner: document)
    monkeypatch.setattr(socket.socket, "connect", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("plan attempted network access")
    ))
    plan = plan_modal_sft_recipe(RECIPE, project_root=tmp_path, profiles_root=PROFILES)
    assert plan.prepared_identity.revision == semantic.dataset_digest
    assert plan.workload_digest
    assert plan.to_dict()["resource_request"] == {
        "accelerator": "A100-80GB", "accelerator_count": 1,
        "timeout_seconds": 1800,
    }
    assert plan.to_dict()["operator_maximum_cost"] == {
        "currency": "USD", "minor_units": 200,
        "semantics": "operator-maximum-not-provider-billing-cap",
    }


def test_train_quote_requires_explicit_scope_without_importing_provider(capsys):
    handler = TrainHandler(args=Namespace(modal_profile=None, modal_environment=None, json=True))
    assert handler._quote_job_config(object()) == 2
    assert "MODAL_QUOTE_SCOPE_REQUIRED" in capsys.readouterr().out


def test_train_quote_reports_only_validated_scoped_gpu_rates(monkeypatch, capsys):
    import tuner.training.modal_host_runtime as runtime
    import tuner.training.modal_host_scope as scope

    client = object()
    binding = object()
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace())
    monkeypatch.setattr(scope, "open_modal_host_scope", lambda **_kwargs: (client, binding))
    monkeypatch.setattr(
        runtime, "observe_scoped_modal_gpu_rates",
        lambda *, sdk, client, client_binding: {"gpu_hour_cost_a100_80gb": "2.00"}
        if client_binding is binding else {},
    )
    handler = TrainHandler(args=Namespace(
        modal_profile="named", modal_environment="existing", json=True,
    ))
    plan = SimpleNamespace(to_dict=lambda: {"resource_request": {
        "accelerator": "A100-80GB", "accelerator_count": 1, "timeout_seconds": 1800,
    }})
    assert handler._quote_job_config(plan) == 0
    output = capsys.readouterr().out
    assert "gpu_hour_cost_a100_80gb" in output
    assert "read-only observation" in output
    assert "token" not in output.lower()
