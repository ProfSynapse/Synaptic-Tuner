"""Provider-free proofs for env-backed GRPO at the recipe, compile and boundary level.

Env-GRPO is accepted by the public contract, the method-keyed Modal recipe
loader, packaged compilation and the host packaged boundary. No checked-in
runtime profile, release or worker admits it yet, so every launch path must
fail closed before planning.
"""

from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from synaptic_tuner.api.v1._contract import PreparedTrainingInputIdentity
from synaptic_tuner.api.v1.training_input import (
    EnvGRPOHyperparametersV1, TrainingInputV1, TrainingMethodV1,
)
from tests.contract.test_public_training_input_v1 import _grpo_document
from tests.runtime.test_packaged_runtime_releases import _provider, _release
from tests.training.test_packaged_execution_material import packaged_fixture
from tuner.handlers.train_handler import TrainHandler
from tuner.runtime.releases import PackagedExecutionBindingV1
from tuner.training.contracts import (
    ArtifactPolicy, CanonicalDocument, ResolvedTrainingComponents, ResourceSpec,
    RuntimeSpec, TrainingRequest,
)
from tuner.training.modal_recipe import (
    ModalMethodNotLaunchableError, ModalRecipeV1, ModalSFTRecipePlanV1,
    load_modal_recipe, plan_modal_sft_recipe,
)
from tuner.training.packaged_boundary import validate_packaged_material
from tuner.training.packaged_compilation import (
    ENV_GRPO_ROLLOUT_LOG_PATH, ENV_ROLLOUT_ROW_FORMAT, PACKAGED_CONTEXT_SCHEMA,
    PACKAGED_ENTRYPOINT, PACKAGED_ENV_GRPO_CONFIG_SCHEMA, PACKAGED_ENV_GRPO_ENTRYPOINT,
    PACKAGED_ENV_GRPO_WORKLOAD_SCHEMA, PACKAGED_SFT_WORKLOAD_SCHEMA,
    compile_packaged_env_grpo_workload, compile_packaged_sft_workload,
    compile_packaged_workload, packaged_artifact_policy_digest,
    packaged_configuration_digest,
)


ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / "Trainers" / "runtime_profiles"
SFT_RECIPE = ROOT / "Trainers/recipes/qwen35_4b_32k_modal_prompt_completion.yaml"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SIX_ROLES = ["final_model", "rollout_log", "tokenizer", "training_lineage",
             "training_metrics", "workload_record"]


def _grpo_recipe() -> dict[str, object]:
    return {
        "name": "qwen35-4b-env-grpo-fixture",
        "target": "cloud",
        "method": "grpo",
        "provider": "modal",
        "job": {
            "runtime_profile": "qwen35-grpo-fixture",
            "accelerator": "L40S",
            "accelerator_count": 1,
            "timeout_seconds": 1800,
            "maximum_cost_minor_units": 300,
        },
        "run": {"method": "grpo", "dry_run": False},
        "model": {"name": "Qwen/Qwen3.5-4B", "revision": REVISION, "load_in_4bit": False},
        "dataset": {
            "local_file": ".tracking/datasets/dataset-" + "d" * 64 + "/dataset.jsonl",
            "schema_version": ENV_ROLLOUT_ROW_FORMAT,
            "expected_digest": "d" * 64,
            "expected_split_counts": {"train": 16, "validation": 0},
        },
        "training": {
            "batch_size": 1, "gradient_accumulation": 4, "learning_rate": 5.0e-6,
            "max_steps": 10, "num_generations": 4, "max_completion_length": 220,
            "temperature": 0.6, "beta": 0.04, "save_steps": 10, "save_total_limit": 1,
            "use_vllm": False, "allow_transformers_rollout_func": True,
        },
        "env_training": {
            "env_backend": "local", "max_turns": 6, "max_tool_steps": 8,
            "token_faithful": True, "context_token_policy": "mask",
        },
        "rewards": {
            "config_ref": "rewards://syntunia/env-grpo-default/v1",
            "config_digest": "c" * 64,
        },
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05,
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        "artifacts": {"required_kinds": list(SIX_ROLES), "retain_checkpoints": False},
    }


@pytest.fixture
def grpo_profiles(tmp_path: Path) -> Path:
    """A test-only profile declaring grpo; no checked-in GRPO profile exists yet."""
    profiles = tmp_path / "Trainers" / "runtime_profiles"
    profiles.mkdir(parents=True)
    document = yaml.safe_load((PROFILES / "qwen35-sft-v1.yaml").read_text(encoding="utf-8"))
    document["name"] = "qwen35-grpo-fixture"
    document["compatibility"]["methods"] = ["grpo"]
    document["runtime"].pop("packaged_build_profile")
    (profiles / "qwen35-grpo-fixture.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    (profiles / "qwen35-sft-v1.inventory.json").write_bytes(
        (PROFILES / "qwen35-sft-v1.inventory.json").read_bytes()
    )
    return profiles


def _load(monkeypatch, profiles: Path, mutation=lambda _document: None) -> ModalRecipeV1:
    from tuner.training import modal_recipe

    document = deepcopy(_grpo_recipe())
    mutation(document)
    monkeypatch.setattr(modal_recipe, "load_recipe", lambda _path, _runner: document)
    return load_modal_recipe(SFT_RECIPE, profiles_root=profiles)


def _identity(digest: str = "d" * 64, row_format: str = ENV_ROLLOUT_ROW_FORMAT):
    return PreparedTrainingInputIdentity(
        "prepared://sha256/" + digest, digest, "e" * 64, 4096, row_format,
    )


def _config(monkeypatch, profiles) -> CanonicalDocument:
    return _load(monkeypatch, profiles).packaged_config(_identity())


# --- Method-keyed recipe loader -------------------------------------------


def test_grpo_recipe_maps_to_public_and_packaged_controls(monkeypatch, grpo_profiles):
    recipe = _load(monkeypatch, grpo_profiles)
    assert recipe.method is TrainingMethodV1.GRPO
    assert type(recipe.hyperparameters) is EnvGRPOHyperparametersV1
    assert recipe.hyperparameters.env_backend == "local"
    assert recipe.hyperparameters.lora_target_modules == ("k_proj", "o_proj", "q_proj", "v_proj")
    assert recipe.artifacts.required_kinds == tuple(SIX_ROLES)
    assert recipe.post_training is None
    public = recipe.training_input(_identity().ref)
    assert public.method is TrainingMethodV1.GRPO
    assert TrainingInputV1.from_json(public.canonical_json()) == public
    config = recipe.packaged_config(_identity()).to_dict()
    assert config["schema_version"] == PACKAGED_ENV_GRPO_CONFIG_SCHEMA
    assert config["method"] == "grpo"
    assert config["grpo"] == public.hyperparameters.to_dict()
    assert "sft" not in config and "post_training" not in config
    assert config["dataset"]["format"] == ENV_ROLLOUT_ROW_FORMAT


def test_sft_recipe_still_loads_with_sft_method():
    recipe = load_modal_recipe(SFT_RECIPE, profiles_root=PROFILES)
    assert recipe.method is TrainingMethodV1.SFT
    assert recipe.packaged_config(PreparedTrainingInputIdentity(
        "prepared://sha256/" + recipe.dataset_digest, recipe.dataset_digest,
        "a" * 64, 123, "syntunia-sft-row/v2",
    )).to_dict()["method"] == "sft"


def test_grpo_recipe_requires_profile_declaring_grpo(monkeypatch):
    """The checked-in SFT profile does not admit grpo, so no real recipe loads yet."""
    with pytest.raises(ValueError, match="does not support method 'grpo'"):
        _load(monkeypatch, PROFILES,
              lambda data: data["job"].update(runtime_profile="qwen35-sft-v1"))


@pytest.mark.parametrize("mutation", [
    lambda data: data["env_training"].update(env_backend="e2b"),
    lambda data: data["env_training"].update(e2b_api_key="secret"),
    lambda data: data["env_training"].update(runtime={"python_packages": ["trl"]}),
    lambda data: data["env_training"].pop("max_turns"),
    lambda data: data["dataset"].update(dataset_name="org/hub-dataset"),
    lambda data: data["dataset"].update(dataset_file="rollouts.jsonl"),
    lambda data: data["dataset"].pop("local_file"),
    lambda data: data["dataset"].update(local_file="/abs/dataset.jsonl"),
    lambda data: data["dataset"].update(local_file="../outside/dataset.jsonl"),
    lambda data: data["dataset"].update(schema_version="syntunia-sft-row/v2"),
    lambda data: data["dataset"].update(format="messages"),
    lambda data: data["training"].update(use_vllm=True),
    lambda data: data["training"].update(allow_transformers_rollout_func=False),
    lambda data: data["training"].update(vllm_mode="colocate"),
    lambda data: data["training"].update(num_epochs=1),
    lambda data: data["training"].pop("num_generations"),
    lambda data: data["training"].update(num_generations=1),
    lambda data: data["training"].update(lerning_rate=1e-5),
    lambda data: data["rewards"].update(success_reward=1.5),
    lambda data: data["rewards"].update(config_digest="not-a-digest"),
    lambda data: data["lora"].update(use_dora=True),
    lambda data: data["model"].update(max_seq_length=4096),
    lambda data: data["run"].update(method="sft"),
    lambda data: data["run"].update(dry_run=True),
    lambda data: data["job"].update(accelerator_count=2),
    lambda data: data["artifacts"].update(required_kinds=SIX_ROLES[:1] + SIX_ROLES[2:]),
    lambda data: data["artifacts"].update(required_kinds=["final_model"]),
    lambda data: data.update(post_training={"evaluation": {}}),
    lambda data: data.update(method="kto"),
])
def test_grpo_recipe_rejects_unknown_hostile_or_incompatible_controls(
    monkeypatch, grpo_profiles, mutation,
):
    with pytest.raises((TypeError, ValueError)):
        _load(monkeypatch, grpo_profiles, mutation)


def test_sft_recipe_rejects_grpo_only_sections(monkeypatch):
    from tuner.training import modal_recipe

    document = yaml.safe_load(SFT_RECIPE.read_text(encoding="utf-8"))
    document["env_training"] = _grpo_recipe()["env_training"]
    monkeypatch.setattr(modal_recipe, "load_recipe", lambda _path, _runner: document)
    with pytest.raises(ValueError, match="do not apply to method sft"):
        load_modal_recipe(SFT_RECIPE, profiles_root=PROFILES)


def test_grpo_recipe_cannot_be_planned_or_launched(monkeypatch, grpo_profiles, tmp_path, capsys):
    from tuner.training import modal_recipe

    document = _grpo_recipe()
    monkeypatch.setattr(modal_recipe, "load_recipe", lambda _path, _runner: document)
    with pytest.raises(ModalMethodNotLaunchableError, match="no runtime profile"):
        plan_modal_sft_recipe(SFT_RECIPE, project_root=tmp_path, profiles_root=grpo_profiles)
    recipe = load_modal_recipe(SFT_RECIPE, profiles_root=grpo_profiles)
    with pytest.raises(ModalMethodNotLaunchableError):
        ModalSFTRecipePlanV1(recipe, _identity(), "sha256:" + "0" * 64, "image",
                             "0" * 64, "0" * 64, "0" * 64)

    monkeypatch.setattr(modal_recipe, "plan_modal_sft_recipe", lambda *_a, **_k: (
        _ for _ in ()).throw(ModalMethodNotLaunchableError("not launchable")))
    recipe_path = tmp_path / "grpo.yaml"
    recipe_path.write_text("name: fixture\n", encoding="utf-8")
    handler = TrainHandler(args=Namespace(job_config=str(recipe_path), plan=True, json=True))
    assert handler.handle() == 2
    assert "MODAL_METHOD_NOT_LAUNCHABLE" in capsys.readouterr().out


# --- Packaged compilation -------------------------------------------------


def test_env_grpo_workload_declares_six_roles_and_rollout_log(monkeypatch, grpo_profiles):
    config = _config(monkeypatch, grpo_profiles)
    workload = compile_packaged_env_grpo_workload(resolved_config=config)
    assert workload == compile_packaged_workload(resolved_config=config)
    document = workload.document
    assert (workload.method, workload.schema_version, workload.entrypoint) == (
        "grpo", PACKAGED_ENV_GRPO_WORKLOAD_SCHEMA, PACKAGED_ENV_GRPO_ENTRYPOINT,
    )
    roles = [item["role"] for item in document["artifacts"]["requirements"]]
    assert sorted(roles) == SIX_ROLES
    assert all(item["minimum"] == item["maximum"] == 1
               for item in document["artifacts"]["requirements"])
    assert document["artifacts"]["schema_version"] == "synaptic-env-grpo-artifacts/v1"
    assert document["artifacts"]["rollout_log"] == {
        "path": ENV_GRPO_ROLLOUT_LOG_PATH,
        "record_schema": "synaptic-env-grpo-rollout-record/v1",
        "required_record_fields": ["prefix_mismatch_count"],
    }
    assert ENV_GRPO_ROLLOUT_LOG_PATH == "logs/rollouts.jsonl"
    assert document["runtime_requirements"]["environment_backend"] == "local"
    assert document["runtime_requirements"]["offline_trainer"] is True
    assert document["identities"]["dataset"]["format"] == ENV_ROLLOUT_ROW_FORMAT
    assert document["configuration"]["digest"] == packaged_configuration_digest(config)


def test_env_grpo_workload_is_deterministic_and_distinct_from_sft(monkeypatch, grpo_profiles):
    config = _config(monkeypatch, grpo_profiles)
    reverse = CanonicalDocument.from_mapping(dict(reversed(list(config.to_dict().items()))))
    assert compile_packaged_env_grpo_workload(resolved_config=config) == \
        compile_packaged_env_grpo_workload(resolved_config=reverse)
    with pytest.raises(ValueError):
        compile_packaged_sft_workload(resolved_config=config)
    _, sft_components = packaged_fixture()
    with pytest.raises(ValueError):
        compile_packaged_env_grpo_workload(resolved_config=sft_components.resolved_config)
    assert compile_packaged_workload(resolved_config=sft_components.resolved_config) == \
        compile_packaged_sft_workload(resolved_config=sft_components.resolved_config)


def _mutated(config: CanonicalDocument, mutation) -> CanonicalDocument:
    value = json.loads(config.canonical_json)
    mutation(value)
    return CanonicalDocument.from_mapping(value)


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(unknown=True),
    lambda value: value.update(post_training={"evaluation": {}}),
    lambda value: value.update(sft=value["grpo"]),
    lambda value: value.pop("grpo"),
    lambda value: value.update(method="sft"),
    lambda value: value.update(method="kto"),
    lambda value: value.update(schema_version="synaptic-packaged-sft-config/v1"),
    lambda value: value.update(execution={"mode": "developer_integration"}),
    lambda value: value["grpo"].update(unknown=1),
    lambda value: value["grpo"].update(env_backend="e2b"),
    lambda value: value["grpo"].update(use_vllm=True),
    lambda value: value["grpo"].update(lora_target_modules=["v_proj", "k_proj"]),
    lambda value: value["grpo"].update(beta=-0.0),
    lambda value: value["grpo"].update(lora_dropout=-0.0),
    lambda value: value["dataset"].update(format="syntunia-sft-row/v2"),
    lambda value: value["dataset"].update(ref="hf://org/dataset"),
    lambda value: value["dataset"].update(dataset_name="org/dataset"),
    lambda value: value.update(dataset={"local_file": "rollouts.jsonl"}),
    lambda value: value["model"].update(revision="main"),
    lambda value: value["model"].update(unknown=True),
])
def test_env_grpo_compile_rejects_unknown_hostile_or_non_local_inputs(
    monkeypatch, grpo_profiles, mutation,
):
    config = _mutated(_config(monkeypatch, grpo_profiles), mutation)
    with pytest.raises((TypeError, ValueError)):
        compile_packaged_env_grpo_workload(resolved_config=config)
    with pytest.raises((TypeError, ValueError)):
        compile_packaged_workload(resolved_config=config)


def test_recipe_rejects_prepared_identity_of_other_format(monkeypatch, grpo_profiles):
    recipe = _load(monkeypatch, grpo_profiles)
    with pytest.raises(ValueError, match="differs from recipe identity"):
        recipe.packaged_config(_identity(row_format="syntunia-sft-row/v2"))


# --- Packaged host boundary -----------------------------------------------


def _grpo_material(monkeypatch, grpo_profiles, **release_changes):
    recipe = _load(monkeypatch, grpo_profiles)
    identity = _identity()
    public = recipe.training_input(identity.ref)
    config = recipe.packaged_config(identity)
    release_values = dict(
        worker_entrypoint=PACKAGED_ENV_GRPO_ENTRYPOINT,
        workload_schema=PACKAGED_ENV_GRPO_WORKLOAD_SCHEMA,
        compatible_methods=("grpo",),
        compatible_models=(("Qwen/Qwen3.5-4B", REVISION),),
        compatible_dataset_formats=(ENV_ROLLOUT_ROW_FORMAT,),
        artifact_contract_schema="synaptic-env-grpo-artifacts/v1",
    )
    release_values.update(release_changes)
    release = _release(**release_values)
    policy = recipe.artifact_policy()
    compiled = compile_packaged_workload(resolved_config=config)
    binding = PackagedExecutionBindingV1.build(
        run_ref="run-env-grpo", runtime_release=release,
        provider_runtime_binding=_provider(release),
        **identity.execution_binding_fields(), workload_digest=compiled.fingerprint,
        configuration_digest=packaged_configuration_digest(config),
        artifact_policy_digest=packaged_artifact_policy_digest(policy),
    )
    components = ResolvedTrainingComponents(
        binding, CanonicalDocument.from_mapping({"schema_version": PACKAGED_CONTEXT_SCHEMA,
                                                "runtime_release": release.to_dict()}),
        config, RuntimeSpec(release.image_ref, release.installed_distributions_digest,
                            release.python_version), ResourceSpec("cpu"), policy,
    )
    return TrainingRequest(CanonicalDocument(public.canonical_json())), components, compiled


def test_boundary_admits_env_grpo_material_under_exact_release(monkeypatch, grpo_profiles):
    request, components, compiled = _grpo_material(monkeypatch, grpo_profiles)
    assert validate_packaged_material(request=request, material=components) == compiled


@pytest.mark.parametrize("changes", [
    {"compatible_methods": ("sft",)},
    {"worker_entrypoint": PACKAGED_ENTRYPOINT},
    {"workload_schema": PACKAGED_SFT_WORKLOAD_SCHEMA},
    {"compatible_dataset_formats": ("syntunia-sft-row/v2",)},
    {"artifact_contract_schema": "synaptic-sft-artifacts/v1"},
    {"compatible_models": (("Qwen/Qwen3.5-4B", "0" * 40),)},
])
def test_boundary_rejects_release_that_does_not_admit_env_grpo(
    monkeypatch, grpo_profiles, changes,
):
    request, components, _ = _grpo_material(monkeypatch, grpo_profiles, **changes)
    with pytest.raises(ValueError):
        validate_packaged_material(request=request, material=components)


def test_boundary_rejects_request_that_differs_from_grpo_configuration(
    monkeypatch, grpo_profiles,
):
    _, components, _ = _grpo_material(monkeypatch, grpo_profiles)
    document = _grpo_document()
    document["model"] = {"ref": "Qwen/Qwen3.5-4B", "revision": REVISION,
                         "tokenizer_revision": REVISION}
    document["dataset"] = {"ref": _identity().ref}
    document["hyperparameters"]["temperature"] = 0.9  # type: ignore[index]
    request = TrainingRequest(CanonicalDocument(
        TrainingInputV1.from_dict(document).canonical_json()
    ))
    with pytest.raises(ValueError):
        validate_packaged_material(request=request, material=components)


def test_sft_boundary_rejects_policy_requiring_grpo_only_role():
    training_input, components = packaged_fixture()
    policy = ArtifactPolicy(("final_model", "rollout_log"), False)
    material = ResolvedTrainingComponents(
        components.execution_source, components.execution_context,
        components.resolved_config, components.runtime, components.resources, policy,
    )
    with pytest.raises(ValueError):
        validate_packaged_material(
            request=TrainingRequest(CanonicalDocument(training_input.canonical_json())),
            material=material,
        )


def test_policy_digest_rejects_roles_outside_every_method_contract():
    with pytest.raises(ValueError):
        packaged_artifact_policy_digest(ArtifactPolicy(("final_model", "checkpoints"), False))
    assert packaged_artifact_policy_digest(ArtifactPolicy(tuple(SIX_ROLES), False))
