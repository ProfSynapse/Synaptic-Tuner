"""Closed mapping from an existing training recipe to packaged Modal SFT inputs."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from types import MappingProxyType

from synaptic_tuner.api.v1._contract import PreparedTrainingInputIdentity
from synaptic_tuner.api.v1.training_input import (
    SFTTrainingHyperparametersV1, TrainingArtifactRequirementsV1,
    TrainingDatasetInputV1, TrainingDurationV1, TrainingInputV1,
    TrainingMethodV1, TrainingModelInputV1,
)
from tuner.discovery.recipes import load_recipe
from tuner.runtime_profiles import load_runtime_profile
from tuner.training.contracts import ArtifactPolicy, CanonicalDocument
from tuner.training.packaged_compilation import PACKAGED_SFT_CONFIG_SCHEMA
from tuner.training.packaged_compilation import (
    compile_packaged_sft_workload, packaged_configuration_digest,
)


_PROFILE = "qwen35-sft-v1"
_ROLES = ("final_model", "tokenizer", "training_lineage", "training_metrics", "workload_record")
MODAL_SFT_ACCELERATOR_RATE_KEYS = MappingProxyType({
    "A100-80GB": "gpu_hour_cost_a100_80gb",
    "L40S": "gpu_hour_cost_l40s",
})


def _section(value: object, allowed: set[str], name: str) -> dict:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be an object")
    if set(value) - allowed:
        raise ValueError(f"{name} contains unsupported fields")
    return value


@dataclass(frozen=True, slots=True)
class ModalSFTRecipeV1:
    name: str
    dataset_locator: str
    dataset_digest: str
    train_rows: int
    validation_rows: int
    runtime_profile: str
    model: TrainingModelInputV1
    hyperparameters: SFTTrainingHyperparametersV1
    artifacts: TrainingArtifactRequirementsV1
    accelerator: str
    accelerator_count: int
    timeout_seconds: int
    maximum_cost_minor_units: int | None

    def training_input(self, dataset_ref: str) -> TrainingInputV1:
        return TrainingInputV1(
            "synaptic-training-input/v1", TrainingMethodV1.SFT, self.model,
            TrainingDatasetInputV1(dataset_ref), self.hyperparameters, self.artifacts,
        )

    def packaged_config(self, identity: PreparedTrainingInputIdentity) -> CanonicalDocument:
        if type(identity) is not PreparedTrainingInputIdentity:
            raise TypeError("exact prepared identity required")
        if identity.format != "syntunia-sft-row/v2" or identity.revision != self.dataset_digest:
            raise ValueError("prepared input differs from recipe identity")
        public = self.training_input(identity.ref)
        return CanonicalDocument.from_mapping({
            "schema_version": PACKAGED_SFT_CONFIG_SCHEMA,
            "method": "sft",
            "execution": {"mode": "packaged_runtime"},
            "model": {**self.model.to_dict(), "load_in_4bit": False},
            "dataset": identity.to_dict(),
            "sft": public.hyperparameters.to_dict(),
        })

    def artifact_policy(self) -> ArtifactPolicy:
        return ArtifactPolicy(self.artifacts.required_kinds, self.artifacts.retain_checkpoints)


@dataclass(frozen=True, slots=True)
class ModalSFTRecipePlanV1:
    recipe: ModalSFTRecipeV1
    prepared_identity: PreparedTrainingInputIdentity
    profile_sha256: str
    profile_base_image: str
    runtime_material_intent_digest: str
    workload_digest: str
    configuration_digest: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "synaptic-modal-sft-recipe-plan/v1",
            "runtime_profile": self.recipe.runtime_profile,
            "profile_sha256": self.profile_sha256,
            "profile_base_image": self.profile_base_image,
            "runtime_material_intent_digest": self.runtime_material_intent_digest,
            "model": self.recipe.model.to_dict(),
            "prepared_dataset": self.prepared_identity.to_dict(),
            "private_preparation": "pending" if os.name == "posix" else "required-at-execution",
            "split_counts": {
                "train": self.recipe.train_rows,
                "validation": self.recipe.validation_rows,
            },
            "configuration_digest": self.configuration_digest,
            "workload_digest": self.workload_digest,
            "artifact_policy": {
                "required_kinds": list(self.recipe.artifacts.required_kinds),
                "retain_checkpoints": self.recipe.artifacts.retain_checkpoints,
            },
            "resource_request": {
                "accelerator": self.recipe.accelerator,
                "accelerator_count": self.recipe.accelerator_count,
                "timeout_seconds": self.recipe.timeout_seconds,
            },
            "operator_maximum_cost": None if self.recipe.maximum_cost_minor_units is None else {
                "currency": "USD",
                "minor_units": self.recipe.maximum_cost_minor_units,
                "semantics": "operator-maximum-not-provider-billing-cap",
            },
        }


def plan_modal_sft_recipe(recipe_path: Path, *, project_root: Path,
                          profiles_root: Path) -> ModalSFTRecipePlanV1:
    """Resolve and verify one publication without any provider or host mutation."""
    from tuner.dataset_prep.publication import snapshot_prepared_dataset_v2

    recipe = load_modal_sft_recipe(recipe_path, profiles_root=profiles_root)
    root = (project_root / ".tracking" / "datasets").resolve(strict=True)
    locator = project_root / recipe.dataset_locator
    directory = locator.parent if locator.name == "dataset.jsonl" else locator
    directory = directory.resolve(strict=True)
    if directory.parent != root:
        raise ValueError("dataset locator is outside the authorized prepared root")
    if os.name == "posix":
        from tuner.training.modal_host_prepared_copy import inspect_mounted_prepared_dataset

        verified = inspect_mounted_prepared_dataset(directory)
    else:
        verified, _ = snapshot_prepared_dataset_v2(directory)
    semantic = verified.semantic_identity
    if (semantic.dataset_digest != recipe.dataset_digest
            or dict(semantic.split_counts) != {
                "train": recipe.train_rows, "validation": recipe.validation_rows,
            }):
        raise ValueError("prepared publication differs from recipe assertions")
    identity = PreparedTrainingInputIdentity(
        f"prepared://sha256/{semantic.dataset_digest}", semantic.dataset_digest,
        semantic.dataset_sha256, semantic.dataset_bytes, "syntunia-sft-row/v2",
    )
    config = recipe.packaged_config(identity)
    workload = compile_packaged_sft_workload(resolved_config=config)
    profile = load_runtime_profile(recipe.runtime_profile, profiles_root)
    from tuner.execution.providers.modal.runtime_build import plan_modal_build_material

    build_profile = (profiles_root.parent / "image_profiles"
                     / "qwen35_4b_packaged_sft_3360351c" / "profile.yaml")
    build_intent = plan_modal_build_material(build_profile)
    if build_intent["base_image"].removeprefix("docker.io/") != profile.image.removeprefix("docker.io/"):
        raise ValueError("Modal build intent differs from the admitted runtime profile")
    return ModalSFTRecipePlanV1(
        recipe, identity, profile.profile_sha256, profile.image,
        build_intent["intent_digest"],
        workload.fingerprint, packaged_configuration_digest(config),
    )


def load_modal_sft_recipe(path: Path, *, profiles_root: Path) -> ModalSFTRecipeV1:
    """Load the existing YAML shape, rejecting provider authority and unknown controls."""
    cfg = _section(load_recipe(path, "cloud"), {
        "name", "description", "target", "method", "provider", "job", "run",
        "model", "dataset", "training", "lora", "artifacts",
    }, "recipe")
    if cfg.get("target") != "cloud" or cfg.get("provider") != "modal" or cfg.get("method") != "sft":
        raise ValueError("Modal SFT requires target: cloud, provider: modal, method: sft")
    job = _section(cfg.get("job"), {
        "runtime_profile", "accelerator", "accelerator_count", "timeout_seconds",
        "maximum_cost_minor_units",
    }, "job")
    if job.get("runtime_profile") != _PROFILE:
        raise ValueError("runtime profile is not admitted for Modal SFT")
    if (type(job.get("accelerator")) is not str
            or job["accelerator"] not in MODAL_SFT_ACCELERATOR_RATE_KEYS
            or type(job.get("accelerator_count")) is not int
            or job["accelerator_count"] != 1
            or type(job.get("timeout_seconds")) is not int
            or not 1 <= job["timeout_seconds"] <= 3600):
        raise ValueError("Modal Qwen SFT requires one supported bounded GPU request")
    maximum_cost = job.get("maximum_cost_minor_units")
    if maximum_cost is not None and (type(maximum_cost) is not int or maximum_cost < 1):
        raise ValueError("operator maximum cost must be positive USD minor units")
    run = _section(cfg.get("run"), {"method", "dry_run"}, "run")
    if run.get("method") != "sft" or run.get("dry_run") is not False:
        raise ValueError("Modal SFT requires an explicit real optimization run")
    model = _section(cfg.get("model"), {"name", "revision", "max_seq_length", "load_in_4bit"}, "model")
    if model.get("load_in_4bit") is not False:
        raise ValueError("this profile requires non-quantized model loading")
    profile = load_runtime_profile(_PROFILE, profiles_root).resolve(
        model=model.get("name"), model_revision=model.get("revision"), method="sft",
    )
    if profile is None:
        raise ValueError("runtime profile did not resolve")
    model_input = TrainingModelInputV1(model["name"], model["revision"], model["revision"])
    dataset = _section(cfg.get("dataset"), {
        "local_file", "schema_version", "format", "use_preassigned_splits",
        "split_dataset", "expected_digest", "expected_split_counts",
    }, "dataset")
    if dataset.get("schema_version") != "syntunia-sft-row/v2" or dataset.get("format") != "messages":
        raise ValueError("Modal SFT requires a prepared v2 messages dataset")
    if dataset.get("use_preassigned_splits") is not True or dataset.get("split_dataset") is not False:
        raise ValueError("Modal SFT requires preassigned splits")
    counts = _section(dataset.get("expected_split_counts"), {"train", "validation"}, "expected_split_counts")
    if set(counts) != {"train", "validation"}:
        raise ValueError("both prepared split counts are required")
    locator = dataset.get("local_file")
    if type(locator) is not str or not locator or Path(locator).is_absolute() or ".." in Path(locator).parts:
        raise ValueError("dataset.local_file must be a safe relative locator")
    training = _section(cfg.get("training"), {
        "batch_size", "gradient_accumulation", "learning_rate", "num_epochs", "max_steps",
        "packing", "completion_only_loss", "assistant_only_loss", "prompt_render",
        "require_memory_efficient_loss", "save_steps", "save_total_limit", "seed",
    }, "training")
    if ("num_epochs" in training) == ("max_steps" in training):
        raise ValueError("training requires exactly one duration")
    lora = _section(cfg.get("lora"), {
        "r", "alpha", "dropout", "target_modules", "use_dora", "use_rslora",
        "init_lora_weights",
    }, "lora")
    artifact_cfg = _section(cfg.get("artifacts"), {"required_kinds", "retain_checkpoints"}, "artifacts")
    kinds = tuple(sorted(artifact_cfg.get("required_kinds", _ROLES)))
    if kinds != tuple(sorted(_ROLES)):
        raise ValueError("Modal SFT requires the complete five-artifact inventory")
    artifacts = TrainingArtifactRequirementsV1(kinds, artifact_cfg.get("retain_checkpoints", False))
    hyperparameters = SFTTrainingHyperparametersV1(
        batch_size=training["batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation"],
        learning_rate=training["learning_rate"],
        duration=TrainingDurationV1(training.get("max_steps"), training.get("num_epochs")),
        max_seq_length=model["max_seq_length"], seed=training.get("seed", 42),
        save_steps=training["save_steps"], save_total_limit=training["save_total_limit"],
        lora_rank=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        lora_target_modules=tuple(sorted(lora["target_modules"])),
        use_dora=lora.get("use_dora", False), use_rslora=lora.get("use_rslora", False),
        init_lora_weights=lora.get("init_lora_weights", True),
        split_dataset=False, dataset_format="messages",
        completion_only_loss=training["completion_only_loss"],
        assistant_only_loss=training["assistant_only_loss"],
        use_preassigned_splits=True, prompt_render=training["prompt_render"],
        packing=training["packing"],
        require_memory_efficient_loss=training["require_memory_efficient_loss"],
    )
    return ModalSFTRecipeV1(
        cfg.get("name", path.stem), locator, dataset["expected_digest"],
        counts["train"], counts["validation"], _PROFILE, model_input,
        hyperparameters, artifacts, job["accelerator"],
        job["accelerator_count"], job["timeout_seconds"], maximum_cost,
    )
