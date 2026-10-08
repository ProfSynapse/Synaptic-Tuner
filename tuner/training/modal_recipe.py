"""Closed mapping from an existing training recipe to packaged Modal inputs.

The recipe is keyed by ``method``: SFT and env-backed GRPO share the job,
model and artifact sections and differ in their dataset and training controls.
Only SFT can be planned and launched today; env-GRPO is accepted at the
contract and compile level and has a pinned runtime profile
(``qwen35-env-grpo-v1``), but no prepared-dataset publisher, worker or Modal
dispatch yet.
"""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import os
from pathlib import Path
from types import MappingProxyType

from synaptic_tuner.api.v1._contract import PreparedTrainingInputIdentity
from synaptic_tuner.api.v1.training_input import (
    EnvGRPOHyperparametersV1, SFTTrainingHyperparametersV1,
    TrainingArtifactRequirementsV1, TrainingDatasetInputV1, TrainingDurationV1,
    TrainingInputV1, TrainingMethodV1, TrainingModelInputV1,
)
from tuner.discovery.recipes import load_recipe
from tuner.runtime_profiles import RuntimeProfile, load_runtime_profile
from tuner.training.contracts import ArtifactPolicy, CanonicalDocument
from tuner.training.packaged_compilation import (
    ENV_ROLLOUT_ROW_FORMAT, PACKAGED_ENV_GRPO_CONFIG_SCHEMA, PACKAGED_SFT_CONFIG_SCHEMA,
    compile_packaged_sft_workload, packaged_configuration_digest,
)
from tuner.training.post_training import validate_post_training_config


_SFT_ROW_FORMAT = "syntunia-sft-row/v2"
_ROLES = ("final_model", "tokenizer", "training_lineage", "training_metrics", "workload_record")
_METHOD_ROLES = MappingProxyType({
    TrainingMethodV1.SFT: _ROLES,
    TrainingMethodV1.GRPO: _ROLES + ("rollout_log",),
})
_METHOD_ROW_FORMATS = MappingProxyType({
    TrainingMethodV1.SFT: _SFT_ROW_FORMAT,
    TrainingMethodV1.GRPO: ENV_ROLLOUT_ROW_FORMAT,
})
MODAL_ACCELERATOR_RATE_KEYS = MappingProxyType({
    "A100-80GB": "gpu_hour_cost_a100_80gb",
    "L40S": "gpu_hour_cost_l40s",
})


def _section(value: object, allowed: set[str], name: str) -> dict:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be an object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        named = []
        for key in unknown:
            close = difflib.get_close_matches(key, sorted(allowed), n=1)
            named.append(f"{name}.{key}" + (f" (did you mean '{close[0]}'?)" if close else ""))
        raise ValueError(f"{name} contains unsupported fields: {', '.join(named)}")
    return value


class ModalMethodNotLaunchableError(ValueError):
    """The recipe is valid, but its method has no admitted Modal runtime yet."""


@dataclass(frozen=True, slots=True)
class ModalRecipeV1:
    name: str
    method: TrainingMethodV1
    dataset_locator: str
    dataset_digest: str
    train_rows: int
    validation_rows: int
    runtime_profile: str
    model: TrainingModelInputV1
    hyperparameters: SFTTrainingHyperparametersV1 | EnvGRPOHyperparametersV1
    artifacts: TrainingArtifactRequirementsV1
    accelerator: str
    accelerator_count: int
    timeout_seconds: int
    maximum_cost_minor_units: int | None
    post_training: dict[str, object] | None = None

    def training_input(self, dataset_ref: str) -> TrainingInputV1:
        return TrainingInputV1(
            "synaptic-training-input/v1", self.method, self.model,
            TrainingDatasetInputV1(dataset_ref), self.hyperparameters, self.artifacts,
        )

    def packaged_config(self, identity: PreparedTrainingInputIdentity) -> CanonicalDocument:
        if type(identity) is not PreparedTrainingInputIdentity:
            raise TypeError("exact prepared identity required")
        if (identity.format != _METHOD_ROW_FORMATS[self.method]
                or identity.revision != self.dataset_digest):
            raise ValueError("prepared input differs from recipe identity")
        public = self.training_input(identity.ref)
        schema, section = {
            TrainingMethodV1.SFT: (PACKAGED_SFT_CONFIG_SCHEMA, "sft"),
            TrainingMethodV1.GRPO: (PACKAGED_ENV_GRPO_CONFIG_SCHEMA, "grpo"),
        }[self.method]
        document = {
            "schema_version": schema,
            "method": self.method.value,
            "execution": {"mode": "packaged_runtime"},
            "model": {**self.model.to_dict(), "load_in_4bit": False},
            "dataset": identity.to_dict(),
            section: public.hyperparameters.to_dict(),
        }
        if self.post_training is not None:
            document["post_training"] = validate_post_training_config(self.post_training)
        return CanonicalDocument.from_mapping(document)

    def artifact_policy(self) -> ArtifactPolicy:
        return ArtifactPolicy(self.artifacts.required_kinds, self.artifacts.retain_checkpoints)


@dataclass(frozen=True, slots=True)
class ModalSFTRecipePlanV1:
    recipe: ModalRecipeV1
    prepared_identity: PreparedTrainingInputIdentity
    profile_sha256: str
    profile_base_image: str
    runtime_material_intent_digest: str
    workload_digest: str
    configuration_digest: str

    def __post_init__(self) -> None:
        if type(self.recipe) is not ModalRecipeV1 or self.recipe.method is not TrainingMethodV1.SFT:
            raise ModalMethodNotLaunchableError("only SFT recipes can be planned for Modal")

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

    recipe = load_modal_recipe(recipe_path, profiles_root=profiles_root)
    if recipe.method is not TrainingMethodV1.SFT:
        raise ModalMethodNotLaunchableError(
            f"Modal {recipe.method.value} is accepted at contract/compile level and has a "
            "pinned runtime profile, but no prepared-dataset publisher, worker or dispatch yet"
        )
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
        semantic.dataset_sha256, semantic.dataset_bytes, _SFT_ROW_FORMAT,
    )
    config = recipe.packaged_config(identity)
    workload = compile_packaged_sft_workload(resolved_config=config)
    profile = load_runtime_profile(recipe.runtime_profile, profiles_root)
    _build_profile, build_intent = resolve_modal_sft_build(profile, profiles_root,
                                                         recipe.model.ref, recipe.model.revision)
    return ModalSFTRecipePlanV1(
        recipe, identity, profile.profile_sha256, profile.image,
        build_intent["intent_digest"],
        workload.fingerprint, packaged_configuration_digest(config),
    )


def resolve_modal_sft_build(profile: RuntimeProfile, profiles_root: Path,
                            model: str, revision: str) -> tuple[Path, dict[str, object]]:
    """Bind a named runtime profile to one compatible packaged build intent."""
    from tuner.cloud.derived_training_image import load_profile
    from tuner.execution.providers.modal.runtime_build import plan_modal_build_material
    from tuner.training.packaged_compilation import PACKAGED_SFT_WORKLOAD_SCHEMA

    build_path = profile.modal_build_profile_path(profiles_root.parent / "image_profiles")
    intent = plan_modal_build_material(build_path)
    build = load_profile(build_path)
    if (intent["profile_digest"] != build.canonical_sha256
            or intent["base_image"].removeprefix("docker.io/") != profile.image.removeprefix("docker.io/")
            or build.packaged_runtime is None):
        raise ValueError("Modal build intent differs from the admitted runtime profile")
    compatibility = build.packaged_runtime["capabilities"]["compatibility"]
    contracts = build.packaged_runtime["capabilities"]["contracts"]
    if ("sft" not in compatibility["methods"]
            or {"ref": model, "revision": revision} not in compatibility["models"]
            or _SFT_ROW_FORMAT not in compatibility["dataset_formats"]
            or contracts["workload_schema"] != PACKAGED_SFT_WORKLOAD_SCHEMA):
        raise ValueError("Modal build capability differs from the admitted runtime profile")
    return build_path, intent


def load_modal_recipe(path: Path, *, profiles_root: Path) -> ModalRecipeV1:
    """Load the existing YAML shape, rejecting provider authority and unknown controls."""
    cfg = _section(load_recipe(path, "cloud"), {
        "name", "description", "target", "method", "provider", "job", "run",
        "model", "dataset", "training", "lora", "artifacts", "post_training",
        "env_training", "rewards",
    }, "recipe")
    method = next((item for item in TrainingMethodV1 if item.value == cfg.get("method")), None)
    if cfg.get("target") != "cloud" or cfg.get("provider") != "modal" or method is None:
        raise ValueError(
            "Modal recipes require target: cloud, provider: modal and method: "
            + " or ".join(item.value for item in TrainingMethodV1)
        )
    method_sections = {TrainingMethodV1.SFT: {"post_training"},
                       TrainingMethodV1.GRPO: {"env_training", "rewards"}}
    foreign = sorted(set().union(*method_sections.values()) - method_sections[method] & set(cfg))
    if foreign:
        raise ValueError(f"recipe sections {', '.join(foreign)} do not apply to method {method.value}")
    job = _section(cfg.get("job"), {
        "runtime_profile", "accelerator", "accelerator_count", "timeout_seconds",
        "maximum_cost_minor_units",
    }, "job")
    profile_name = job.get("runtime_profile")
    if type(profile_name) is not str:
        raise ValueError("Modal training requires a named runtime profile")
    if (type(job.get("accelerator")) is not str
            or job["accelerator"] not in MODAL_ACCELERATOR_RATE_KEYS
            or type(job.get("accelerator_count")) is not int
            or job["accelerator_count"] != 1
            or type(job.get("timeout_seconds")) is not int
            or not 1 <= job["timeout_seconds"] <= 86400):
        raise ValueError("Modal training requires one supported bounded GPU request")
    maximum_cost = job.get("maximum_cost_minor_units")
    if maximum_cost is not None and (type(maximum_cost) is not int or maximum_cost < 1):
        raise ValueError("operator maximum cost must be positive USD minor units")
    run = _section(cfg.get("run"), {"method", "dry_run"}, "run")
    if run.get("method") != method.value or run.get("dry_run") is not False:
        raise ValueError("Modal training requires an explicit real optimization run")
    model_fields = {"name", "revision", "load_in_4bit"}
    if method is TrainingMethodV1.SFT:
        model_fields.add("max_seq_length")
    model = _section(cfg.get("model"), model_fields, "model")
    if model.get("load_in_4bit") is not False:
        raise ValueError("this profile requires non-quantized model loading")
    profile = load_runtime_profile(profile_name, profiles_root).resolve(
        model=model.get("name"), model_revision=model.get("revision"), method=method.value,
    )
    if profile is None:
        raise ValueError("runtime profile did not resolve")
    model_input = TrainingModelInputV1(model["name"], model["revision"], model["revision"])
    dataset_fields = {"local_file", "schema_version", "expected_digest", "expected_split_counts"}
    if method is TrainingMethodV1.SFT:
        dataset_fields |= {"format", "use_preassigned_splits", "split_dataset"}
    dataset = _section(cfg.get("dataset"), dataset_fields, "dataset")
    if dataset.get("schema_version") != _METHOD_ROW_FORMATS[method]:
        raise ValueError(
            f"Modal {method.value} requires prepared {_METHOD_ROW_FORMATS[method]} rows"
        )
    if method is TrainingMethodV1.SFT:
        if dataset.get("format") != "messages":
            raise ValueError("Modal SFT requires a prepared v2 messages dataset")
        if dataset.get("use_preassigned_splits") is not True or dataset.get("split_dataset") is not False:
            raise ValueError("Modal SFT requires preassigned splits")
    counts = _section(dataset.get("expected_split_counts"), {"train", "validation"}, "expected_split_counts")
    if set(counts) != {"train", "validation"}:
        raise ValueError("both prepared split counts are required")
    # The dataset is always a prepared offline input; there is no hub download path.
    locator = dataset.get("local_file")
    if type(locator) is not str or not locator or Path(locator).is_absolute() or ".." in Path(locator).parts:
        raise ValueError("dataset.local_file must be a safe relative locator")
    artifact_cfg = _section(cfg.get("artifacts"), {"required_kinds", "retain_checkpoints"}, "artifacts")
    roles = _METHOD_ROLES[method]
    kinds = tuple(sorted(artifact_cfg.get("required_kinds", roles)))
    if kinds != tuple(sorted(roles)):
        raise ValueError(f"Modal {method.value} requires the complete {len(roles)}-artifact inventory")
    artifacts = TrainingArtifactRequirementsV1(kinds, artifact_cfg.get("retain_checkpoints", False))
    if method is TrainingMethodV1.SFT:
        hyperparameters = _sft_hyperparameters(cfg, model)
        post_training = validate_post_training_config(cfg.get("post_training"))
    else:
        hyperparameters = _env_grpo_hyperparameters(cfg)
        post_training = None
    return ModalRecipeV1(
        cfg.get("name", path.stem), method, locator, dataset["expected_digest"],
        counts["train"], counts["validation"], profile_name, model_input,
        hyperparameters, artifacts, job["accelerator"],
        job["accelerator_count"], job["timeout_seconds"], maximum_cost, post_training,
    )


def _sft_hyperparameters(cfg: dict, model: dict) -> SFTTrainingHyperparametersV1:
    training = _section(cfg.get("training"), {
        "batch_size", "gradient_accumulation", "learning_rate", "num_epochs", "max_steps",
        "packing", "completion_only_loss", "assistant_only_loss", "prompt_render",
        "require_memory_efficient_loss", "chat_template_kwargs", "save_steps", "save_total_limit", "seed",
    }, "training")
    if ("num_epochs" in training) == ("max_steps" in training):
        raise ValueError("training requires exactly one duration")
    lora = _section(cfg.get("lora"), {
        "r", "alpha", "dropout", "target_modules", "use_dora", "use_rslora",
        "init_lora_weights",
    }, "lora")
    return SFTTrainingHyperparametersV1(
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
        chat_template_kwargs=training.get("chat_template_kwargs"),
    )


def _required(section: dict, keys: tuple[str, ...], name: str) -> None:
    missing = [key for key in keys if key not in section]
    if missing:
        raise ValueError(f"{name} requires: {', '.join(f'{name}.{key}' for key in missing)}")


def _env_grpo_hyperparameters(cfg: dict) -> EnvGRPOHyperparametersV1:
    training_keys = (
        "batch_size", "gradient_accumulation", "learning_rate", "max_steps",
        "num_generations", "max_completion_length", "temperature", "beta",
        "save_steps", "save_total_limit", "use_vllm", "allow_transformers_rollout_func",
    )
    training = _section(cfg.get("training"), set(training_keys) | {"seed"}, "training")
    _required(training, training_keys, "training")
    env_keys = ("env_backend", "max_turns", "max_tool_steps", "token_faithful", "context_token_policy")
    env = _section(cfg.get("env_training"), set(env_keys), "env_training")
    _required(env, env_keys, "env_training")
    rewards = _section(cfg.get("rewards"), {"config_ref", "config_digest"}, "rewards")
    _required(rewards, ("config_ref", "config_digest"), "rewards")
    lora_keys = ("r", "alpha", "dropout", "target_modules")
    lora = _section(cfg.get("lora"), set(lora_keys), "lora")
    _required(lora, lora_keys, "lora")
    if type(lora["target_modules"]) is not list:
        raise ValueError("lora.target_modules must be a list")
    return EnvGRPOHyperparametersV1(
        batch_size=training["batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation"],
        learning_rate=training["learning_rate"], max_steps=training["max_steps"],
        seed=training.get("seed", 42), save_steps=training["save_steps"],
        save_total_limit=training["save_total_limit"],
        num_generations=training["num_generations"],
        max_completion_length=training["max_completion_length"],
        temperature=training["temperature"], beta=training["beta"],
        lora_rank=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        lora_target_modules=tuple(sorted(lora["target_modules"])),
        env_backend=env["env_backend"], max_turns=env["max_turns"],
        max_tool_steps=env["max_tool_steps"], token_faithful=env["token_faithful"],
        context_token_policy=env["context_token_policy"],
        reward_config_ref=rewards["config_ref"], reward_config_digest=rewards["config_digest"],
        use_vllm=training["use_vllm"],
        allow_transformers_rollout_func=training["allow_transformers_rollout_func"],
    )
