"""Pure retained-input GRPO projection; this module grants no execution authority."""
from __future__ import annotations

import hashlib
import re

from synaptic_tuner.api.v1.training_input import (
    GRPOTrainingHyperparametersV1, TrainingInputV1, TrainingMethodV1,
    validate_grpo_rewards,
)
from synaptic_tuner.api.v1.training_sources import PreparedTrainingInputIdentity
from .recipes import CompiledWorkload, canonical_json_bytes

GRPO_CONFIG_SCHEMA = "synaptic-grpo-config/v1"
GRPO_WORKLOAD_SCHEMA = "synaptic-grpo-workload/v1"
GRPO_PREPARED_FORMAT = "synaptic-grpo-prompts/v1"
GRPO_ENTRYPOINT = "Trainers/grpo/runtime_v1.py"
_ROLES = frozenset({"workload_record", "training_lineage", "training_metrics", "final_model", "tokenizer"})


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def grpo_prepared_semantic_digest(*, content_digest, size_bytes, row_count, prompt_column):
    basis = {"schema_version": "synaptic-grpo-prepared-publication/v1",
             "format": GRPO_PREPARED_FORMAT, "prompt_column": prompt_column,
             "content_digest": content_digest, "size_bytes": size_bytes, "row_count": row_count}
    return _sha(b"synaptic-grpo-prepared-input/v1\0" + canonical_json_bytes(basis))


def normalized_grpo_configuration(training_input: TrainingInputV1,
                                  prepared: PreparedTrainingInputIdentity) -> dict:
    """Project exact declarations without reading model, dataset or reward paths."""
    if type(training_input) is not TrainingInputV1 or training_input.method is not TrainingMethodV1.GRPO:
        raise TypeError("exact GRPO training input required")
    value = TrainingInputV1.from_dict(training_input.to_dict())
    if type(prepared) is not PreparedTrainingInputIdentity or prepared.format != GRPO_PREPARED_FORMAT:
        raise ValueError("GRPO prepared format required")
    if value.dataset.ref != prepared.ref or not 1 <= prepared.size_bytes <= 65536:
        raise ValueError("GRPO prepared identity differs from input")
    if any(re.fullmatch(r"[0-9a-f]{40}", revision) is None
           for revision in (value.model.revision, value.model.tokenizer_revision)):
        raise ValueError("GRPO requires immutable model and tokenizer revisions")
    if value.artifacts.retain_checkpoints or set(value.artifacts.required_kinds) != _ROLES:
        raise ValueError("GRPO requires the five native roles without checkpoints")
    hp = value.hyperparameters
    assert type(hp) is GRPOTrainingHyperparametersV1
    rewards = validate_grpo_rewards(value.rewards)
    training = {
        "output_dir": "checkpoints", "per_device_train_batch_size": hp.batch_size,
        "gradient_accumulation_steps": hp.gradient_accumulation_steps,
        "num_generations": hp.num_generations, "max_steps": hp.max_steps,
        "max_prompt_length": hp.max_prompt_length, "max_completion_length": hp.max_completion_length,
        "learning_rate": hp.learning_rate, "temperature": hp.temperature,
        "beta": hp.beta, "weight_decay": hp.weight_decay, "warmup_ratio": hp.warmup_ratio,
        "max_grad_norm": hp.max_grad_norm, "logging_steps": hp.logging_steps,
        "save_steps": hp.save_steps, "save_total_limit": hp.save_total_limit,
        "fp16": hp.fp16, "bf16": hp.bf16, "optim": "adamw_8bit",
        "lr_scheduler_type": "linear", "num_train_epochs": 1,
        "report_to": "none", "use_gspo": False, "extra_args": {},
    }
    if hp.chat_template_kwargs is not None:
        training["chat_template_kwargs"] = hp.to_dict()["chat_template_kwargs"]
    return {
        "model": {"model_name": value.model.ref, "lora_path": None,
                  "max_seq_length": hp.max_seq_length, "dtype": None,
                  "load_in_4bit": hp.load_in_4bit, "chat_template": "native"},
        "lora": {"r": hp.lora_rank, "lora_alpha": hp.lora_alpha,
                 "lora_dropout": hp.lora_dropout, "bias": "none",
                 "target_modules": list(hp.lora_target_modules),
                 "use_gradient_checkpointing": "unsloth", "random_state": hp.seed,
                 "use_rslora": hp.use_rslora, "use_dora": hp.use_dora},
        "training": training,
        "dataset": {"dataset_name": None, "dataset_file": None, "local_file": None,
                    "num_proc": 1, "prompt_column": hp.prompt_column},
        "rewards": {"items": rewards["items"], "custom": {"enabled": False}},
        "pivot": {"enabled": False}, "wandb": {"enabled": False}, "seed": hp.seed,
    }


def compile_grpo_input(training_input: TrainingInputV1,
                       prepared: PreparedTrainingInputIdentity) -> CompiledWorkload:
    if type(training_input) is not TrainingInputV1:
        raise TypeError("exact GRPO training input required")
    value = TrainingInputV1.from_dict(training_input.to_dict())
    config = normalized_grpo_configuration(value, prepared)
    rewards = validate_grpo_rewards(value.rewards)
    model = {"model_id": value.model.ref, "revision": value.model.revision,
             "tokenizer_revision": value.model.tokenizer_revision}
    document = {
        "schema_version": GRPO_WORKLOAD_SCHEMA, "method": "grpo", "entrypoint": GRPO_ENTRYPOINT,
        "configuration_schema": GRPO_CONFIG_SCHEMA, "config": config, "model": model,
        "prepared_input": prepared.to_dict(),
        "reward_documents": rewards["documents"], "reward_schema": rewards["schema"],
        "config_sha256": _sha(canonical_json_bytes(config)),
        "dataset_sha256": prepared.content_digest,
        "rewards_sha256": _sha(canonical_json_bytes({"documents": rewards["documents"], "schema": rewards["schema"]})),
    }
    return CompiledWorkload("grpo", GRPO_WORKLOAD_SCHEMA, GRPO_ENTRYPOINT,
                            canonical_json_bytes(document))


def admit_grpo_snapshot(training_input: TrainingInputV1,
                        prepared: PreparedTrainingInputIdentity, snapshot: bytes):
    """Validate actual rows through the retained core, without opening an execution lease."""
    if type(prepared) is not PreparedTrainingInputIdentity:
        raise TypeError("exact prepared identity required")
    if type(snapshot) is not bytes or len(snapshot) != prepared.size_bytes or _sha(snapshot) != prepared.content_digest:
        raise ValueError("GRPO prepared snapshot differs from identity")
    try:
        text = snapshot.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("GRPO prepared input must be UTF-8") from None
    if text.startswith("\ufeff") or not 1 <= len(snapshot.splitlines()) <= 128:
        raise ValueError("GRPO prepared input BOM or row count is invalid")
    compiled = compile_grpo_input(training_input, prepared).document
    from Trainers.grpo.runtime_v1 import admit_grpo_inputs
    admission = admit_grpo_inputs(config=compiled["config"], dataset_bytes=snapshot,
        reward_documents=compiled["reward_documents"], reward_schema=compiled["reward_schema"],
        model_identity=compiled["model"], workload=compiled)
    if compiled["config"]["dataset"]["prompt_column"] != "prompt" and any(
            "prompt" in row for row in admission[1]):
        raise ValueError("GRPO input collides with derived prompt column")
    semantic = grpo_prepared_semantic_digest(content_digest=prepared.content_digest,
        size_bytes=prepared.size_bytes, row_count=len(admission[1]),
        prompt_column=compiled["config"]["dataset"]["prompt_column"])
    if prepared.revision != semantic or prepared.ref != "prepared://sha256/" + semantic:
        raise ValueError("GRPO prepared semantic identity differs from request")
    return admission
