"""Fixed packaged GRPO projection, separate from developer core workload bytes."""
from __future__ import annotations

from synaptic_tuner.api.v1.training_sources import PreparedTrainingInputIdentity
from synaptic_tuner.api.v1.training_input import TrainingInputV1
from tuner.runtime.releases import parse_packaged_runtime_release
from tuner.training.grpo_compilation import GRPO_WORKLOAD_SCHEMA, GRPO_ENTRYPOINT, compile_grpo_input
from tuner.training.packaged_compilation import PACKAGED_CONTEXT_SCHEMA, PACKAGED_ENTRYPOINT, _domain
from tuner.training.recipes import CompiledWorkload, canonical_json_bytes

PACKAGED_GRPO_WORKLOAD_SCHEMA = "synaptic-packaged-grpo-workload/v1"
PACKAGED_GRPO_ARTIFACT_SCHEMA = "synaptic-grpo-artifacts/v1"
ROLES = ("workload_record", "training_lineage", "training_metrics", "final_model", "tokenizer")
_CORE_FIELDS = {"schema_version", "method", "entrypoint", "configuration_schema", "config",
                "model", "prepared_input", "reward_documents", "reward_schema",
                "config_sha256", "dataset_sha256", "rewards_sha256"}


def _declared_input(document):
    """Invert only declared fields, then require the exact G1 compiler output."""
    config = document["config"]
    training, lora, model = config["training"], config["lora"], config["model"]
    hp = {"schema_version": "synaptic-grpo-hyperparameters/v1",
        "batch_size": training["per_device_train_batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "num_generations": training["num_generations"], "max_steps": training["max_steps"],
        "max_seq_length": model["max_seq_length"], "max_prompt_length": training["max_prompt_length"],
        "max_completion_length": training["max_completion_length"], "learning_rate": training["learning_rate"],
        "temperature": training["temperature"], "beta": training["beta"],
        "weight_decay": training["weight_decay"], "warmup_ratio": training["warmup_ratio"],
        "max_grad_norm": training["max_grad_norm"], "seed": config["seed"],
        "logging_steps": training["logging_steps"], "save_steps": training["save_steps"],
        "save_total_limit": training["save_total_limit"], "lora_rank": lora["r"],
        "lora_alpha": lora["lora_alpha"], "lora_dropout": lora["lora_dropout"],
        "lora_target_modules": lora["target_modules"], "use_dora": lora["use_dora"],
        "use_rslora": lora["use_rslora"], "load_in_4bit": model["load_in_4bit"],
        "bf16": training["bf16"], "fp16": training["fp16"],
        "prompt_column": config["dataset"]["prompt_column"]}
    if "chat_template_kwargs" in training:
        hp["chat_template_kwargs"] = training["chat_template_kwargs"]
    return TrainingInputV1.from_dict({
        "schema_version": "synaptic-training-input/v1", "method": "grpo",
        "model": {"ref": document["model"]["model_id"], "revision": document["model"]["revision"],
                  "tokenizer_revision": document["model"]["tokenizer_revision"]},
        "dataset": {"ref": document["prepared_input"]["ref"]}, "hyperparameters": hp,
        "rewards": {"items": config["rewards"]["items"], "documents": document["reward_documents"],
                    "schema": document["reward_schema"]},
        "artifacts": {"required_kinds": sorted(ROLES), "retain_checkpoints": False},
    })


def _core(document):
    from Trainers.grpo.runtime_v1 import _sha, _canonical, _finite_tree
    if (type(document) is not dict or set(document) != _CORE_FIELDS
            or document["schema_version"] != GRPO_WORKLOAD_SCHEMA
            or document["method"] != "grpo" or document["entrypoint"] != GRPO_ENTRYPOINT
            or document["configuration_schema"] != "synaptic-grpo-config/v1"):
        raise ValueError("PACKAGED_GRPO_CORE_REJECTED")
    _finite_tree(document)
    prepared = PreparedTrainingInputIdentity.from_dict(document["prepared_input"])
    if (prepared.format != "synaptic-grpo-prompts/v1" or prepared.size_bytes > 65536
            or prepared.content_digest != document["dataset_sha256"]
            or _sha(_canonical(document["config"])) != document["config_sha256"]
            or _sha(_canonical({"documents": document["reward_documents"],
                               "schema": document["reward_schema"]})) != document["rewards_sha256"]):
        raise ValueError("PACKAGED_GRPO_CORE_REJECTED")
    declared = _declared_input(document)
    recompiled = compile_grpo_input(declared, prepared)
    if recompiled.canonical_bytes != canonical_json_bytes(document):
        raise ValueError("PACKAGED_GRPO_CORE_REJECTED")
    return recompiled


def compile_packaged_grpo_workload(*, core_workload, context, child_max_seconds,
                                   qualification_evidence=None):
    if type(core_workload) is not CompiledWorkload:
        raise TypeError("exact GRPO core workload required")
    core = _core(core_workload.document)
    if core.canonical_bytes != core_workload.canonical_bytes:
        raise ValueError("PACKAGED_GRPO_CORE_REJECTED")
    if (type(context) is not dict or set(context) != {"schema_version", "runtime_release"}
            or context["schema_version"] != PACKAGED_CONTEXT_SCHEMA):
        raise ValueError("PACKAGED_GRPO_CONTEXT_REJECTED")
    release = parse_packaged_runtime_release(context["runtime_release"])
    if (release.to_dict() != context["runtime_release"]
            or release.worker_entrypoint != PACKAGED_ENTRYPOINT
            or release.prepared_input_schema != "synaptic-prepared-training-input/v1"
            or (core.document["model"]["model_id"], core.document["model"]["revision"]) not in release.compatible_models
            or "grpo" not in release.compatible_methods
            or release.workload_schema != PACKAGED_GRPO_WORKLOAD_SCHEMA
            or "synaptic-grpo-prompts/v1" not in release.compatible_dataset_formats
            or release.artifact_contract_schema != PACKAGED_GRPO_ARTIFACT_SCHEMA):
        raise ValueError("PACKAGED_GRPO_CONTEXT_REJECTED")
    if type(child_max_seconds) is not int or not 1 <= child_max_seconds <= 86400:
        raise ValueError("PACKAGED_GRPO_LIMIT_REJECTED")
    limits = {"child_max_seconds": child_max_seconds}
    if qualification_evidence is not None:
        from Trainers.grpo.runtime_v1 import validate_qualification_evidence_policy
        limits["qualification_evidence"] = validate_qualification_evidence_policy(qualification_evidence)
    document = {
        "schema_version": PACKAGED_GRPO_WORKLOAD_SCHEMA, "method": "grpo",
        "entrypoint": PACKAGED_ENTRYPOINT, "core_workload": core.document,
        "context": context, "limits": limits,
        "artifacts": {"schema_version": PACKAGED_GRPO_ARTIFACT_SCHEMA,
                      "requirements": [{"role": role, "minimum": 1, "maximum": 1}
                                       for role in ROLES]},
    }
    return CompiledWorkload("grpo", PACKAGED_GRPO_WORKLOAD_SCHEMA, PACKAGED_ENTRYPOINT,
                            canonical_json_bytes(document))


def parse_packaged_grpo_workload(raw):
    from tuner.runtime.packaged_sft_execution import _document
    document = _document(raw, 1024 * 1024)
    if (set(document) != {"schema_version", "method", "entrypoint", "core_workload",
                         "context", "limits", "artifacts"}
            or type(document["limits"]) is not dict
            or set(document["limits"]) not in ({"child_max_seconds"},
                {"child_max_seconds", "qualification_evidence"})):
        raise ValueError("PACKAGED_GRPO_WORKLOAD_REJECTED")
    optional = {}
    if "qualification_evidence" in document["limits"]:
        from Trainers.grpo.runtime_v1 import validate_qualification_evidence_policy
        optional["qualification_evidence"] = validate_qualification_evidence_policy(document["limits"]["qualification_evidence"])
    compiled = compile_packaged_grpo_workload(core_workload=_core(document["core_workload"]),
        context=document["context"], child_max_seconds=document["limits"]["child_max_seconds"], **optional)
    if compiled.canonical_bytes != raw:
        raise ValueError("PACKAGED_GRPO_WORKLOAD_REJECTED")
    return compiled


def packaged_grpo_configuration_digest(compiled):
    validated = parse_packaged_grpo_workload(compiled.canonical_bytes)
    document = validated.document
    return _domain("synaptic-packaged-grpo-configuration/v1", {
        "core_workload": document["core_workload"], "context": document["context"],
        "limits": document["limits"],
    })
