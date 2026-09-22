"""Provider-neutral canonical packaged SFT compilation.

This module is imported only on the packaged branch. Workload bytes never
contain their execution binding: compile first, build the binding second,
then the host packaged boundary admits complete material before planning or
coordinator retention.

Admission here is not provider-dispatch authority. Slice E (and each later
adapter) must resolve the committed ProviderRuntimeBindingV1 and invoke
PackagedExecutionBindingV1.validate_bindings(runtime_release, provider_binding)
before dispatch, then authenticate its adapter-owned native facts. Shared
material deliberately retains only the provider binding's digest commitment.
"""

from __future__ import annotations

import hashlib
import math
import re

from synaptic_tuner.api.v1._contract import PreparedTrainingInputIdentity
from synaptic_tuner.api.v1.training_input import (
    SFTTrainingHyperparametersV1, TrainingModelInputV1,
)
from .contracts import ArtifactPolicy, CanonicalDocument
from .recipes import CompiledWorkload, canonical_json_bytes, selected_execution_mode


PACKAGED_SFT_CONFIG_SCHEMA = "synaptic-packaged-sft-config/v1"
PACKAGED_SFT_WORKLOAD_SCHEMA = "synaptic-packaged-sft-workload/v1"
PACKAGED_CONTEXT_SCHEMA = "synaptic-packaged-execution-context/v1"
PACKAGED_ENTRYPOINT = "tuner.runtime.packaged_training_worker:main"
_REVISION = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_CONFIG_FIELDS = {"schema_version", "method", "execution", "model", "dataset", "sft"}
_ARTIFACT_ROLES = (
    "workload_record", "training_lineage", "training_metrics", "final_model", "tokenizer",
)


def _fields(value, fields, name):
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{name} has missing or unknown fields")
    return value


def _domain(name: str, value: dict) -> str:
    return hashlib.sha256(name.encode("ascii") + b"\0" + canonical_json_bytes(value)).hexdigest()


def packaged_configuration_digest(config: CanonicalDocument) -> str:
    _validated_config(config)
    return _domain("synaptic-packaged-training-configuration/v1", config.to_dict())


def packaged_artifact_policy_digest(policy: ArtifactPolicy) -> str:
    if type(policy) is not ArtifactPolicy:
        raise TypeError("exact artifact policy required")
    if not set(policy.required_kinds).issubset(_ARTIFACT_ROLES):
        raise ValueError("artifact policy requires roles absent from compiled contract")
    return _domain("synaptic-coordinator-artifact-policy/v1", {
        "required_kinds": list(policy.required_kinds),
        "retain_checkpoints": policy.retain_checkpoints,
    })


def _validated_config(config: CanonicalDocument):
    if type(config) is not CanonicalDocument:
        raise TypeError("exact resolved configuration required")
    value = _fields(config.to_dict(), _CONFIG_FIELDS, "packaged configuration")
    if value["schema_version"] != PACKAGED_SFT_CONFIG_SCHEMA or value["method"] != "sft":
        raise ValueError("unsupported packaged training configuration")
    if selected_execution_mode(config) != "packaged_runtime":
        raise ValueError("packaged compilation requires packaged_runtime mode")
    model = _fields(value["model"], {"ref", "revision", "tokenizer_revision", "load_in_4bit"}, "model")
    TrainingModelInputV1.from_dict({key: model[key] for key in ("ref", "revision", "tokenizer_revision")})
    if any(type(model[key]) is not str or _REVISION.fullmatch(model[key]) is None
           for key in ("revision", "tokenizer_revision")):
        raise ValueError("packaged model requires immutable revisions")
    if type(model["load_in_4bit"]) is not bool:
        raise ValueError("load_in_4bit must be an exact boolean")
    identity = PreparedTrainingInputIdentity.from_dict(value["dataset"])
    hyperparameters = SFTTrainingHyperparametersV1.from_dict(value["sft"])
    # Python equality aliases 0 with 0.0, and -0.0 with +0.0. Only the exact
    # normalized wire representation is admitted to the digest boundary.
    if (hyperparameters.lora_dropout == 0.0
            and math.copysign(1.0, hyperparameters.lora_dropout) < 0.0):
        raise ValueError("packaged hyperparameters must not contain negative zero")
    if canonical_json_bytes(hyperparameters.to_dict()) != canonical_json_bytes(value["sft"]):
        raise ValueError("packaged hyperparameters must already be canonical")
    # Prepared data controls are explicit; the trainer must not reinterpret its split/format.
    formats = {"syntunia-sft-row/v1": "raw_text", "syntunia-sft-row/v2": "messages"}
    if identity.format not in formats or hyperparameters.dataset_format != formats[identity.format]:
        raise ValueError("prepared format differs from SFT controls")
    if hyperparameters.use_preassigned_splits is not True or hyperparameters.split_dataset:
        raise ValueError("prepared input requires preassigned splits")
    if identity.format == "syntunia-sft-row/v2" and (
        hyperparameters.packing is not False
        or hyperparameters.completion_only_loss is not True
        or hyperparameters.assistant_only_loss is not False
        or hyperparameters.prompt_render != "prompt_completion"
    ):
        raise ValueError("prepared messages require prompt completion controls")
    canonical_json_bytes(value)
    return value, identity


def compile_packaged_sft_workload(*, resolved_config: CanonicalDocument) -> CompiledWorkload:
    """Compile without a binding, so the host can subsequently build one."""
    config, identity = _validated_config(resolved_config)
    document = {
        "schema_version": PACKAGED_SFT_WORKLOAD_SCHEMA,
        "method": "sft",
        "entrypoint": PACKAGED_ENTRYPOINT,
        "configuration": {
            "digest": _domain("synaptic-packaged-training-configuration/v1", config),
            "document": config,
        },
        "identities": {"model": config["model"], "dataset": identity.to_dict()},
        "runtime_requirements": {
            "schema_version": "synaptic-packaged-sft-runtime-requirements/v1",
            "offline_trainer": True,
            "credential_free_trainer": True,
        },
        "artifacts": {
            "schema_version": "synaptic-sft-artifacts/v1",
            "requirements": [{"role": role, "minimum": 1, "maximum": 1}
                             for role in _ARTIFACT_ROLES],
        },
    }
    return CompiledWorkload("sft", PACKAGED_SFT_WORKLOAD_SCHEMA, PACKAGED_ENTRYPOINT,
                            canonical_json_bytes(document))




__all__ = [
    "compile_packaged_sft_workload", "packaged_configuration_digest",
    "packaged_artifact_policy_digest", "PACKAGED_SFT_CONFIG_SCHEMA",
    "PACKAGED_SFT_WORKLOAD_SCHEMA", "PACKAGED_CONTEXT_SCHEMA", "PACKAGED_ENTRYPOINT",
]
