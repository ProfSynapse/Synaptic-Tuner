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
from tuner.training.modal_recipe import (load_modal_sft_recipe, plan_modal_sft_recipe,
                                         resolve_modal_sft_build)
from tuner.runtime_profiles import load_runtime_profile
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
        "L40S", 1, 1800,
    )
    assert recipe.maximum_cost_minor_units == 200
    assert workload.fingerprint


def test_third_named_profile_selects_its_declared_packaged_build(tmp_path, monkeypatch):
    profiles = tmp_path / "Trainers" / "runtime_profiles"
    images = tmp_path / "Trainers" / "image_profiles" / "third_packaged_build"
    profiles.mkdir(parents=True)
    images.mkdir(parents=True)
    document = yaml.safe_load((PROFILES / "qwen35-sft-v1.yaml").read_text(encoding="utf-8"))
    document["name"] = "third-sft-profile"
    document["runtime"]["packaged_build_profile"] = "third_packaged_build"
    (profiles / "third-sft-profile.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    (profiles / "qwen35-sft-v1.inventory.json").write_bytes(
        (PROFILES / "qwen35-sft-v1.inventory.json").read_bytes()
    )
    source_build = ROOT / "Trainers/image_profiles/qwen35_4b_packaged_sft_3360351c/profile.yaml"
    (images / "profile.yaml").write_bytes(source_build.read_bytes())
    from tuner.training import modal_recipe
    recipe_document = deepcopy(yaml.safe_load(RECIPE.read_text(encoding="utf-8")))
    recipe_document["job"]["runtime_profile"] = "third-sft-profile"
    monkeypatch.setattr(modal_recipe, "load_recipe", lambda _path, _runner: recipe_document)
    recipe = load_modal_sft_recipe(RECIPE, profiles_root=profiles)
    assert recipe.runtime_profile == "third-sft-profile"
    profile = load_runtime_profile(recipe.runtime_profile, profiles)
    build_path, intent = resolve_modal_sft_build(profile, profiles, recipe.model.ref,
                                                  recipe.model.revision)
    assert build_path == images / "profile.yaml"
    assert intent["base_image"].removeprefix("docker.io/") == profile.image


@pytest.mark.parametrize("change,match", [
    (lambda build: build.update(base_image=build["base_image"][:-1] +
                                ("0" if build["base_image"][-1] != "0" else "1")),
     "intent differs"),
    (lambda build: build["packaged_runtime"]["capabilities"]["compatibility"].update(
        models=[{"ref": "Qwen/Qwen3.5-4B", "revision": "0" * 40}]), "capability differs"),
    (lambda build: build["packaged_runtime"]["capabilities"]["compatibility"].update(
        methods=["kto"]), "capability differs"),
    (lambda build: build["packaged_runtime"]["capabilities"]["contracts"].update(
        workload_schema="other-workload"), "capability differs"),
])
def test_modal_build_rejects_mismatched_profile(tmp_path, change, match):
    profiles = tmp_path / "Trainers" / "runtime_profiles"
    images = tmp_path / "Trainers" / "image_profiles" / "selected"
    profiles.mkdir(parents=True)
    images.mkdir(parents=True)
    document = yaml.safe_load((PROFILES / "qwen35-sft-v1.yaml").read_text(encoding="utf-8"))
    document["runtime"]["packaged_build_profile"] = "selected"
    (profiles / "qwen35-sft-v1.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    (profiles / "qwen35-sft-v1.inventory.json").write_bytes(
        (PROFILES / "qwen35-sft-v1.inventory.json").read_bytes()
    )
    build = yaml.safe_load((ROOT / "Trainers/image_profiles/qwen35_4b_packaged_sft_3360351c/profile.yaml")
                           .read_text(encoding="utf-8"))
    change(build)
    (images / "profile.yaml").write_text(yaml.safe_dump(build, sort_keys=False), encoding="utf-8")
    admitted = load_runtime_profile("qwen35-sft-v1", profiles)
    with pytest.raises(ValueError, match=match):
        resolve_modal_sft_build(admitted, profiles, "Qwen/Qwen3.5-4B",
                                "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")


@pytest.mark.parametrize("mutation", [
    lambda data: data["job"].update(image="arbitrary/image"),
    lambda data: data["job"].update(runtime_profile="unknown"),
    lambda data: data["job"].update(accelerator_count=2),
    lambda data: data["job"].update(accelerator_count=True),
    lambda data: data["job"].update(accelerator="T4"),
    lambda data: data["job"].update(accelerator="l40s"),
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


def test_recipe_names_unsupported_fields_with_suggestions(monkeypatch):
    with pytest.raises(ValueError) as excinfo:
        _load(monkeypatch, lambda data: data["training"].update(lerning_rate=1e-4))
    assert "training.lerning_rate (did you mean 'learning_rate'?)" in str(excinfo.value)


def test_recipe_accepts_reviewed_a100_option(monkeypatch):
    """The existing accelerator admission remains unchanged."""
    recipe = _load(monkeypatch, lambda data: data["job"].update(accelerator="A100-80GB"))
    assert recipe.accelerator == "A100-80GB"


@pytest.mark.parametrize("timeout", [1, 3600, 7200, 14400, 86400])
def test_recipe_timeout_bound_preserves_legacy_workload_bytes(monkeypatch, timeout):
    baseline = load_modal_sft_recipe(RECIPE, profiles_root=PROFILES)
    requested = _load(monkeypatch, lambda data: data["job"].update(timeout_seconds=timeout))
    identity = PreparedTrainingInputIdentity("prepared://sha256/" + baseline.dataset_digest,
        baseline.dataset_digest, "a" * 64, 123, "syntunia-sft-row/v2")
    original_config = baseline.packaged_config(identity)
    changed_config = requested.packaged_config(identity)
    assert requested.timeout_seconds == timeout
    assert original_config.canonical_json.encode("utf-8") == changed_config.canonical_json.encode("utf-8")
    assert compile_packaged_sft_workload(resolved_config=original_config).fingerprint == \
        compile_packaged_sft_workload(resolved_config=changed_config).fingerprint


@pytest.mark.parametrize("timeout", [None, True, False, "86400", 1.5, 0, -1, 86401])
def test_recipe_timeout_rejects_nonfinite_or_inexact_authority(monkeypatch, timeout):
    with pytest.raises(ValueError):
        _load(monkeypatch, lambda data: data["job"].update(timeout_seconds=timeout))


def test_recipe_timeout_is_required(monkeypatch):
    with pytest.raises(ValueError):
        _load(monkeypatch, lambda data: data["job"].pop("timeout_seconds"))


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


@pytest.mark.parametrize("timeout", [1800, 14400, 86400])
def test_plan_verifies_publication_without_network_or_provider_effects(tmp_path, monkeypatch, timeout):
    _, _, _, dataset_config = _config(tmp_path / "source")
    root = tmp_path / ".tracking" / "datasets"
    try:
        published = prepare_dataset_v2(dataset_config, root)
    except DatasetPublicationUncertainV1:
        published = prepare_dataset_v2(dataset_config, root)
    semantic = published.semantic_identity
    document = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
    document["job"]["timeout_seconds"] = timeout
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
        "accelerator": "L40S", "accelerator_count": 1,
        "timeout_seconds": timeout,
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
