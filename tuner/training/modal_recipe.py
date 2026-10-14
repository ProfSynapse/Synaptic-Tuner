"""Closed mapping from an existing training recipe to packaged Modal SFT inputs."""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import os
import re
from pathlib import Path
from types import MappingProxyType

from synaptic_tuner.api.v1._contract import PreparedTrainingInputIdentity
from synaptic_tuner.api.v1.training_input import (
    SFTTrainingHyperparametersV1, TrainingArtifactRequirementsV1,
    TrainingDatasetInputV1, TrainingDurationV1, TrainingInputV1,
    TrainingMethodV1, TrainingModelInputV1,
)
from tuner.discovery.recipes import load_recipe
from tuner.runtime_profiles import RuntimeProfile, load_runtime_profile
from tuner.training.contracts import ArtifactPolicy, CanonicalDocument
from tuner.training.packaged_compilation import PACKAGED_SFT_CONFIG_SCHEMA
from tuner.training.packaged_compilation import (
    compile_packaged_sft_workload, packaged_configuration_digest,
    validate_observation_capture_config,
)
from tuner.training.post_training import validate_post_training_config


_ROLES = ("final_model", "tokenizer", "training_lineage", "training_metrics", "workload_record")
MODAL_SFT_ACCELERATOR_RATE_KEYS = MappingProxyType({
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
    post_training: dict[str, object] | None = None
    observation_capture: dict[str, object] | None = None
    prepared_format: str = "syntunia-sft-row/v2"
    allow_cancel_reconcile: bool = False

    def training_input(self, dataset_ref: str) -> TrainingInputV1:
        return TrainingInputV1(
            "synaptic-training-input/v1", TrainingMethodV1.SFT, self.model,
            TrainingDatasetInputV1(dataset_ref), self.hyperparameters, self.artifacts,
        )

    def packaged_config(self, identity: PreparedTrainingInputIdentity) -> CanonicalDocument:
        if type(identity) is not PreparedTrainingInputIdentity:
            raise TypeError("exact prepared identity required")
        if identity.format != self.prepared_format or identity.revision != self.dataset_digest:
            raise ValueError("prepared input differs from recipe identity")
        public = self.training_input(identity.ref)
        document = {
            "schema_version": PACKAGED_SFT_CONFIG_SCHEMA,
            "method": "sft",
            "execution": {"mode": "packaged_runtime"},
            "model": {**self.model.to_dict(), "load_in_4bit": False},
            "dataset": identity.to_dict(),
            "sft": public.hyperparameters.to_dict(),
        }
        if self.post_training is not None:
            document["post_training"] = validate_post_training_config(self.post_training)
        if self.observation_capture is not None:
            document["observation_capture"] = validate_observation_capture_config(self.observation_capture)
        return CanonicalDocument.from_mapping(document)

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
            **({"allow_cancel_reconcile": True} if self.recipe.allow_cancel_reconcile else {}),
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


def _prepared_dataset_root(project_root: Path, source_root: Path | None) -> Path:
    from tuner.project import ProjectContext, load_project_manifest

    manifest = project_root / "synaptic.yaml"
    if manifest.exists():
        context = load_project_manifest(manifest).create_context(
            engine_root=source_root or Path(__file__).resolve().parents[2],
            invocation_cwd=project_root,
        )
    else:
        context = ProjectContext.standalone(engine_root=project_root)
    return (context.tracking_root / "datasets").resolve(strict=True)


def plan_modal_sft_recipe(recipe_path: Path, *, project_root: Path,
                          profiles_root: Path, source_root: Path | None = None) -> ModalSFTRecipePlanV1:
    """Resolve and verify one publication without any provider or host mutation."""
    from tuner.dataset_prep.publication import snapshot_prepared_dataset_v2

    recipe = load_modal_sft_recipe(recipe_path, profiles_root=profiles_root)
    root = _prepared_dataset_root(project_root, source_root)
    locator = project_root / recipe.dataset_locator
    directory = locator.parent if locator.name == "dataset.jsonl" else locator
    directory = directory.resolve(strict=True)
    if directory.parent != root:
        raise ValueError("dataset locator is outside the authorized prepared root")
    if os.name == "posix":
        from tuner.training.modal_host_prepared_copy import inspect_mounted_prepared_dataset

        verified = inspect_mounted_prepared_dataset(directory, prepared_format=recipe.prepared_format)
    else:
        if recipe.prepared_format == "synaptic-conversation-sft-row/v1":
            from tuner.dataset_prep.conversation_messages import snapshot_conversation_dataset
            verified, _ = snapshot_conversation_dataset(directory)
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
        semantic.dataset_sha256, semantic.dataset_bytes, recipe.prepared_format,
    )
    config = recipe.packaged_config(identity)
    workload = compile_packaged_sft_workload(resolved_config=config)
    profile = load_runtime_profile(recipe.runtime_profile, profiles_root)
    _build_profile, build_intent = resolve_modal_sft_build(profile, profiles_root,
                                                         recipe.model.ref, recipe.model.revision, prepared_format=recipe.prepared_format, source_root=source_root)
    return ModalSFTRecipePlanV1(
        recipe, identity, profile.profile_sha256, profile.image,
        build_intent["intent_digest"],
        workload.fingerprint, packaged_configuration_digest(config),
    )


def resolve_modal_sft_build(profile: RuntimeProfile, profiles_root: Path,
                            model: str, revision: str, *,
                            prepared_format: str = "syntunia-sft-row/v2",
                            source_root: Path | None = None) -> tuple[Path, dict[str, object]]:
    """Bind a named runtime profile to one compatible packaged build intent."""
    from tuner.cloud.derived_training_image import load_profile
    from tuner.execution.providers.modal.runtime_build import plan_modal_build_material
    from tuner.training.packaged_compilation import PACKAGED_SFT_WORKLOAD_SCHEMA

    build_path = profile.modal_build_profile_path(profiles_root.parent / "image_profiles")
    intent = plan_modal_build_material(build_path, **({"source_root": source_root} if source_root is not None else {}))
    build = load_profile(build_path)
    if (intent["profile_digest"] != build.canonical_sha256
            or intent["base_image"].removeprefix("docker.io/") != profile.image.removeprefix("docker.io/")
            or build.packaged_runtime is None):
        raise ValueError("Modal build intent differs from the admitted runtime profile")
    compatibility = build.packaged_runtime["capabilities"]["compatibility"]
    contracts = build.packaged_runtime["capabilities"]["contracts"]
    if ("sft" not in compatibility["methods"]
            or {"ref": model, "revision": revision} not in compatibility["models"]
            or prepared_format not in compatibility["dataset_formats"]
            or contracts["workload_schema"] != PACKAGED_SFT_WORKLOAD_SCHEMA):
        raise ValueError("Modal build capability differs from the admitted runtime profile")
    return build_path, intent



def _cancel_selection(config):
    value = config.get("allow_cancel_reconcile", False)
    if type(value) is not bool:
        raise ValueError("exact cancellation capability selection required")
    return value

def load_modal_sft_recipe(path: Path, *, profiles_root: Path) -> ModalSFTRecipeV1:
    """Load the existing YAML shape, rejecting provider authority and unknown controls."""
    cfg = _section(load_recipe(path, "cloud"), {
        "name", "description", "target", "method", "provider", "job", "run",
        "model", "dataset", "training", "lora", "artifacts", "post_training", "observation_capture", "allow_cancel_reconcile",
    }, "recipe")
    if cfg.get("target") != "cloud" or cfg.get("provider") != "modal" or cfg.get("method") != "sft":
        raise ValueError("Modal SFT requires target: cloud, provider: modal, method: sft")
    job = _section(cfg.get("job"), {
        "runtime_profile", "accelerator", "accelerator_count", "timeout_seconds",
        "maximum_cost_minor_units",
    }, "job")
    profile_name = job.get("runtime_profile")
    if type(profile_name) is not str:
        raise ValueError("Modal SFT requires a named runtime profile")
    if (type(job.get("accelerator")) is not str
            or job["accelerator"] not in MODAL_SFT_ACCELERATOR_RATE_KEYS
            or type(job.get("accelerator_count")) is not int
            or job["accelerator_count"] != 1
            or type(job.get("timeout_seconds")) is not int
            or not 1 <= job["timeout_seconds"] <= 86400):
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
    profile = load_runtime_profile(profile_name, profiles_root).resolve(
        model=model.get("name"), model_revision=model.get("revision"), method="sft",
    )
    if profile is None:
        raise ValueError("runtime profile did not resolve")
    model_input = TrainingModelInputV1(model["name"], model["revision"], model["revision"])
    dataset = _section(cfg.get("dataset"), {
        "local_file", "schema_version", "format", "use_preassigned_splits",
        "split_dataset", "expected_digest", "expected_split_counts",
    }, "dataset")
    prepared_format = dataset.get("schema_version")
    if prepared_format not in {"syntunia-sft-row/v2", "synaptic-conversation-sft-row/v1"} or dataset.get("format") != "messages":
        raise ValueError("Modal SFT requires an explicitly supported prepared messages dataset")
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
        "require_memory_efficient_loss", "chat_template_kwargs", "save_steps", "save_total_limit", "seed",
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
        chat_template_kwargs=training.get("chat_template_kwargs"),
    )
    if prepared_format == "synaptic-conversation-sft-row/v1" and hyperparameters.require_memory_efficient_loss is not True:
        raise ValueError("conversation SFT requires the memory-efficient loss guard")
    return ModalSFTRecipeV1(
        cfg.get("name", path.stem), locator, dataset["expected_digest"],
        counts["train"], counts["validation"], profile_name, model_input,
        hyperparameters, artifacts, job["accelerator"],
        job["accelerator_count"], job["timeout_seconds"], maximum_cost,
        validate_post_training_config(cfg.get("post_training")),
        (validate_observation_capture_config(cfg["observation_capture"])
         if "observation_capture" in cfg else None),
        prepared_format,
        _cancel_selection(cfg),
    )


@dataclass(frozen=True, slots=True)
class ModalGRPORecipeV1:
    """Internal candidate recipe, not a public execution capability."""
    name: str
    dataset_locator: str
    dataset_digest: str
    row_count: int
    runtime_profile: str
    declaration: TrainingInputV1
    accelerator: str
    accelerator_count: int
    timeout_seconds: int
    maximum_cost_minor_units: int | None
    child_max_seconds: int
    prepared_format: str = "synaptic-grpo-prompts/v1"
    qualification_evidence: object = None
    allow_cancel_reconcile: bool = False

    def limits(self):
        value = {"child_max_seconds": self.child_max_seconds}
        if self.qualification_evidence is not None:
            value["qualification_evidence"] = dict(self.qualification_evidence)
        return value

    @property
    def model(self):
        return self.declaration.model

    def training_input(self, dataset_ref):
        if dataset_ref != self.declaration.dataset.ref:
            raise ValueError("GRPO prepared ref differs from recipe")
        return TrainingInputV1.from_dict(self.declaration.to_dict())

    def artifact_policy(self):
        return ArtifactPolicy(self.declaration.artifacts.required_kinds,
                              self.declaration.artifacts.retain_checkpoints)

    def packaged_config(self, identity):
        from tuner.training.grpo_compilation import compile_grpo_input
        if (type(identity) is not PreparedTrainingInputIdentity
                or identity.format != self.prepared_format or identity.revision != self.dataset_digest):
            raise ValueError("GRPO prepared identity differs from recipe")
        core = compile_grpo_input(self.training_input(identity.ref), identity)
        return CanonicalDocument.from_mapping({
            "schema_version": "synaptic-packaged-grpo-config/v1", "method": "grpo",
            "execution": {"mode": "packaged_runtime"}, "core_workload": core.document,
            "limits": self.limits(),
        })


@dataclass(frozen=True, slots=True)
class ModalGRPORecipePlanV1:
    """Core-only preplan; a final workload needs an actual qualified release."""
    recipe: ModalGRPORecipeV1
    prepared_identity: PreparedTrainingInputIdentity
    profile_sha256: str
    profile_base_image: str
    runtime_material_intent_digest: str
    core_workload_digest: str

    def to_dict(self):
        return {
            "schema_version": "synaptic-modal-grpo-recipe-plan/v1", "method": "grpo",
            **({"allow_cancel_reconcile": True} if self.recipe.allow_cancel_reconcile else {}),
            "runtime_profile": self.recipe.runtime_profile, "profile_sha256": self.profile_sha256,
            "profile_base_image": self.profile_base_image,
            "runtime_material_intent_digest": self.runtime_material_intent_digest,
            "training_input": self.recipe.declaration.to_dict(),
            "prepared_dataset": self.prepared_identity.to_dict(), "row_count": self.recipe.row_count,
            "core_workload_digest": self.core_workload_digest,
            "limits": self.recipe.limits(),
            "execution_status": "runtime-binding-and-cpu-qualification-required",
            "resource_request": {"accelerator": self.recipe.accelerator,
                "accelerator_count": self.recipe.accelerator_count, "timeout_seconds": self.recipe.timeout_seconds},
            "operator_maximum_cost": None if self.recipe.maximum_cost_minor_units is None else {
                "currency": "USD", "minor_units": self.recipe.maximum_cost_minor_units,
                "semantics": "operator-maximum-not-provider-billing-cap"},
        }


def load_modal_grpo_recipe(path: Path, *, profiles_root: Path) -> ModalGRPORecipeV1:
    """Consume exact existing public GRPO fields; no reward directory fallback."""
    cfg = _section(load_recipe(path, "cloud"), {
        "name", "description", "target", "provider", "method", "job", "run",
        "dataset", "training_input", "limits", "allow_cancel_reconcile",
    }, "GRPO recipe")
    run = _section(cfg.get("run"), {"method", "dry_run"}, "GRPO run")
    if (cfg.get("target") != "cloud" or cfg.get("provider") != "modal" or cfg.get("method") != "grpo"
            or set(run) != {"method", "dry_run"} or run["method"] != "grpo" or run["dry_run"] is not False):
        raise ValueError("GRPO requires an explicit real optimization declaration")
    public = TrainingInputV1.from_dict(cfg.get("training_input"))
    if public.method is not TrainingMethodV1.GRPO:
        raise ValueError("GRPO recipe and public method differ")
    if set(public.artifacts.required_kinds) != set(_ROLES) or public.artifacts.retain_checkpoints:
        raise ValueError("GRPO requires the complete five native artifacts without checkpoints")
    dataset = _section(cfg.get("dataset"), {
        "local_file", "schema_version", "expected_digest", "expected_row_count",
    }, "GRPO dataset")
    locator = dataset.get("local_file")
    digest, rows = dataset.get("expected_digest"), dataset.get("expected_row_count")
    if (dataset.get("schema_version") != "synaptic-grpo-prompts/v1"
            or type(locator) is not str or not locator or Path(locator).is_absolute() or ".." in Path(locator).parts
            or public.dataset.ref != "prepared://sha256/" + str(digest)
            or type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(rows) is not int or not 1 <= rows <= 128):
        raise ValueError("GRPO publication assertion differs")
    limits = _section(cfg.get("limits"), {"child_max_seconds", "qualification_evidence"}, "GRPO limits")
    child_limit = limits.get("child_max_seconds")
    if type(child_limit) is not int or not 1 <= child_limit <= 86400:
        raise ValueError("GRPO child budget differs")
    evidence = None
    if "qualification_evidence" in limits:
        from Trainers.grpo.runtime_v1 import validate_qualification_evidence_policy
        evidence = MappingProxyType(validate_qualification_evidence_policy(limits["qualification_evidence"]))
    job = _section(cfg.get("job"), {
        "runtime_profile", "accelerator", "accelerator_count", "timeout_seconds", "maximum_cost_minor_units",
    }, "GRPO job")
    if (type(job.get("runtime_profile")) is not str or type(job.get("accelerator")) is not str
            or job.get("accelerator") not in MODAL_SFT_ACCELERATOR_RATE_KEYS
            or type(job.get("accelerator_count")) is not int or job["accelerator_count"] != 1
            or type(job.get("timeout_seconds")) is not int or not 1 <= job["timeout_seconds"] <= 86400):
        raise ValueError("GRPO requires a named profile and bounded existing resource")
    maximum = job.get("maximum_cost_minor_units")
    if maximum is not None and (type(maximum) is not int or maximum < 1):
        raise ValueError("operator maximum cost must be positive USD minor units")
    load_runtime_profile(job["runtime_profile"], profiles_root).resolve(
        model=public.model.ref, model_revision=public.model.revision, method="grpo")
    return ModalGRPORecipeV1(cfg.get("name", path.stem), locator, digest, rows, job["runtime_profile"],
        public, job["accelerator"], job["accelerator_count"], job["timeout_seconds"], maximum, child_limit,
        qualification_evidence=evidence, allow_cancel_reconcile=_cancel_selection(cfg))


def load_modal_recipe(path, *, profiles_root):
    method = load_recipe(path, "cloud").get("method")
    if method == "grpo":
        return load_modal_grpo_recipe(path, profiles_root=profiles_root)
    return load_modal_sft_recipe(path, profiles_root=profiles_root)


def resolve_modal_recipe_build(profile, profiles_root, model, revision, *, method="sft",
                               prepared_format="syntunia-sft-row/v2", source_root=None):
    if method == "sft":
        return resolve_modal_sft_build(profile, profiles_root, model, revision,
            prepared_format=prepared_format, source_root=source_root)
    if method != "grpo" or prepared_format != "synaptic-grpo-prompts/v1":
        raise ValueError("unsupported Modal recipe method or format")
    from tuner.cloud.derived_training_image import load_profile
    from tuner.execution.providers.modal.runtime_build import plan_modal_build_material
    profile.resolve(model=model, model_revision=revision, method="grpo")
    path = profile.modal_build_profile_path(profiles_root.parent / "image_profiles")
    intent = plan_modal_build_material(path, **({"source_root": source_root} if source_root is not None else {}))
    build = load_profile(path)
    if (build.packaged_runtime is None or intent["profile_digest"] != build.canonical_sha256
            or intent["base_image"].removeprefix("docker.io/") != profile.image.removeprefix("docker.io/")):
        raise ValueError("GRPO runtime build identity differs")
    caps = build.packaged_runtime["capabilities"]
    if (profile.methods != ("grpo",) or caps["compatibility"]["methods"] != ["grpo"]
            or {"ref": model, "revision": revision} not in caps["compatibility"]["models"]
            or prepared_format not in caps["compatibility"]["dataset_formats"]
            or caps["contracts"] != {"workload_schema": "synaptic-packaged-grpo-workload/v1",
                "prepared_input_schema": "synaptic-prepared-training-input/v1",
                "artifact_contract_schema": "synaptic-grpo-artifacts/v1"}):
        raise ValueError("GRPO runtime build compatibility differs")
    return path, intent


def plan_modal_recipe(path, *, project_root, profiles_root, source_root=None):
    recipe = load_modal_recipe(path, profiles_root=profiles_root)
    if type(recipe) is ModalSFTRecipeV1:
        return plan_modal_sft_recipe(path, project_root=project_root, profiles_root=profiles_root, source_root=source_root)
    from tuner.training.modal_host_prepared_copy import inspect_mounted_prepared_dataset, _verified_identity
    from tuner.training.grpo_compilation import admit_grpo_snapshot, compile_grpo_input
    locator = (project_root / recipe.dataset_locator).absolute()
    directory = locator.parent if locator.name == "dataset.jsonl" else locator
    if directory.resolve(strict=True).parent != _prepared_dataset_root(project_root, source_root):
        raise ValueError("GRPO publication outside project dataset root")
    def validate(manifest, raw):
        identity = _verified_identity(manifest, recipe.prepared_format)
        if (identity.revision != recipe.dataset_digest or manifest["row_count"] != recipe.row_count
                or manifest["prompt_column"] != recipe.declaration.hyperparameters.prompt_column):
            raise ValueError("GRPO publication differs from recipe")
        admit_grpo_snapshot(recipe.training_input(identity.ref), identity, raw)
    manifest = inspect_mounted_prepared_dataset(directory, prepared_format=recipe.prepared_format,
                                                snapshot_validator=validate)
    identity = _verified_identity(manifest, recipe.prepared_format)
    core = compile_grpo_input(recipe.training_input(identity.ref), identity)
    profile = load_runtime_profile(recipe.runtime_profile, profiles_root)
    _, intent = resolve_modal_recipe_build(profile, profiles_root, recipe.model.ref, recipe.model.revision,
        method="grpo", prepared_format=recipe.prepared_format, source_root=source_root)
    return ModalGRPORecipePlanV1(recipe, identity, profile.profile_sha256, profile.image,
        intent["intent_digest"], core.fingerprint)
