"""Closed-decoder bridge and strict dispatcher validation for SFT v1."""
from __future__ import annotations

import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath
from typing import Any

from .contracts import (
    ARTIFACT_ROLES_V1, ENTRYPOINT_V1, MASK_CONTRACT_V1, SCHEMA_V1,
    BoundWorkloadV1, ClosedWorkloadDecoderV1, FixtureRuntimeBindingEvidenceV1,
    RuntimeContractError,
)

_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
_PROFILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_TARGET = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,127}")


def _mapping(value: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RuntimeContractError(f"{name} has invalid fields")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise RuntimeContractError(f"{name} is not an exact digest")
    return value


def _revision(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HEX40.fullmatch(value) is None:
        raise RuntimeContractError(f"{name} is not an immutable revision")
    return value


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeContractError(f"{name} must be positive")
    return value


def _decimal(value: Any, name: str, *, positive: bool = False) -> Decimal:
    if not isinstance(value, str):
        raise RuntimeContractError(f"{name} must be a canonical decimal string")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise RuntimeContractError(f"{name} is invalid") from exc
    if not number.is_finite() or number < 0 or (positive and number <= 0):
        raise RuntimeContractError(f"{name} is outside its range")
    return number


def decode_verified_document_v1(
    canonical_bytes: bytes,
    decoder: ClosedWorkloadDecoderV1,
) -> tuple[BoundWorkloadV1, Mapping[str, Any]]:
    if not isinstance(canonical_bytes, bytes) or not canonical_bytes:
        raise RuntimeContractError("canonical workload bytes are required")
    bound = decoder.decode_and_bind_sft_v1(canonical_bytes)
    if bound.canonical_bytes != canonical_bytes:
        raise RuntimeContractError("decoder changed canonical workload bytes")
    digest = _digest(bound.digest, "workload digest")
    document = _mapping(bound.document, {
        "adapter", "artifacts", "checkpoints", "code", "dataset", "entrypoint",
        "kind", "logging", "model", "objective", "optimization", "provenance",
        "runtime", "schema_version",
    }, "workload")
    if document["schema_version"] != SCHEMA_V1 or document["entrypoint"] != ENTRYPOINT_V1 or document["kind"] != "sft":
        raise RuntimeContractError("fixed SFT v1 dispatcher rejected workload")

    model = _mapping(document["model"], {
        "repository", "revision", "tokenizer_revision", "trust_remote_code", "weights_format"
    }, "model")
    if not isinstance(model["repository"], str) or _REPO.fullmatch(model["repository"]) is None:
        raise RuntimeContractError("model repository must be owner/name")
    _revision(model["revision"], "model revision")
    _revision(model["tokenizer_revision"], "tokenizer revision")
    if model["trust_remote_code"] is not False or model["weights_format"] != "safetensors":
        raise RuntimeContractError("unsafe model loading policy")

    dataset = _mapping(document["dataset"], {
        "file_selector", "record_format", "repository", "revision", "split"
    }, "dataset")
    if not isinstance(dataset["repository"], str) or _REPO.fullmatch(dataset["repository"]) is None:
        raise RuntimeContractError("dataset repository must be owner/name")
    _revision(dataset["revision"], "dataset revision")
    if dataset["record_format"] != "conversations":
        raise RuntimeContractError("dataset record format is unsupported")
    split = dataset["split"]
    if not isinstance(split, str) or not split or len(split) > 128 or any(ord(c) < 32 or c in "?#\\" for c in split):
        raise RuntimeContractError("dataset split is invalid")
    selector = dataset["file_selector"]
    if not isinstance(selector, str) or not selector or len(selector) > 512 or any(x in selector for x in ("\\", "://", "?", "#", "*", "[", "]", "{", "}")):
        raise RuntimeContractError("dataset selector is invalid")
    path = PurePosixPath(selector)
    if path.is_absolute() or selector != path.as_posix() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeContractError("dataset selector is not canonical")
    if path.suffix.lower() not in {".json", ".jsonl"}:
        raise RuntimeContractError("dataset selector must identify JSON or JSONL")

    adapter = _mapping(document["adapter"], {
        "alpha", "bias", "dropout", "initialization", "rank", "target_modules",
        "target_profile_digest", "target_profile_id", "type", "use_dora", "use_rslora",
    }, "adapter")
    if adapter["type"] != "lora" or adapter["bias"] != "none" or adapter["initialization"] != "default" or adapter["use_dora"] is not False or adapter["use_rslora"] is not False:
        raise RuntimeContractError("adapter policy is unsupported")
    _positive_int(adapter["rank"], "adapter rank")
    _positive_int(adapter["alpha"], "adapter alpha")
    _decimal(adapter["dropout"], "adapter dropout")
    targets = tuple(adapter["target_modules"]) if isinstance(adapter["target_modules"], list) else ()
    if not targets or len(targets) != len(set(targets)) or any(not isinstance(x, str) or _TARGET.fullmatch(x) is None for x in targets):
        raise RuntimeContractError("target profile is invalid")
    if not isinstance(adapter["target_profile_id"], str) or _PROFILE.fullmatch(adapter["target_profile_id"]) is None:
        raise RuntimeContractError("target profile id is invalid")
    _digest(adapter["target_profile_digest"], "target profile digest")

    objective = _mapping(document["objective"], {
        "chat_template", "loss_scope", "masking_contract", "max_seq_length",
        "packing", "prompt_render", "truncation",
    }, "objective")
    if objective["chat_template"] != "tokenizer_embedded_required" or objective["masking_contract"] != MASK_CONTRACT_V1 or objective["packing"] is not False or objective["prompt_render"] != "full_conversation" or objective["truncation"] != "right" or objective["loss_scope"] not in {"assistant_messages", "full_sequence"}:
        raise RuntimeContractError("objective contract is unsupported")
    _positive_int(objective["max_seq_length"], "max sequence length")

    optimization = _mapping(document["optimization"], {
        "batch_size", "dtype", "duration", "gradient_accumulation_steps",
        "gradient_checkpointing", "learning_rate", "max_grad_norm", "model_quantization",
        "optimizer", "scheduler", "seed", "warmup_ratio", "weight_decay",
    }, "optimization")
    if optimization["dtype"] != "bf16" or optimization["model_quantization"] != "none" or optimization["optimizer"] != "adamw_8bit" or optimization["scheduler"] != "linear" or optimization["gradient_checkpointing"] != "unsloth":
        raise RuntimeContractError("optimization runtime is unsupported")
    _positive_int(optimization["batch_size"], "batch size")
    _positive_int(optimization["gradient_accumulation_steps"], "gradient accumulation")
    if not isinstance(optimization["seed"], int) or isinstance(optimization["seed"], bool) or optimization["seed"] < 0:
        raise RuntimeContractError("seed is invalid")
    for field, positive in (("learning_rate", True), ("max_grad_norm", True), ("warmup_ratio", False), ("weight_decay", False)):
        _decimal(optimization[field], field, positive=positive)
    duration = optimization["duration"]
    if not isinstance(duration, Mapping) or set(duration) not in ({"max_steps"}, {"epochs"}):
        raise RuntimeContractError("duration is invalid")
    if "max_steps" in duration:
        _positive_int(duration["max_steps"], "max steps")
    else:
        _decimal(duration["epochs"], "epochs", positive=True)

    artifacts = _mapping(document["artifacts"], {"required_roles"}, "artifacts")
    if tuple(artifacts["required_roles"]) != ARTIFACT_ROLES_V1:
        raise RuntimeContractError("artifact roles do not match runtime contract")
    code = _mapping(document["code"], {"engine_commit", "source_digest"}, "code")
    _revision(code["engine_commit"], "engine commit")
    _digest(code["source_digest"], "source digest")
    runtime = _mapping(document["runtime"], {"dependency_lock_digest"}, "runtime")
    _digest(runtime["dependency_lock_digest"], "dependency lock digest")
    provenance = _mapping(document["provenance"], {
        "class", "fixture_manifest_digest"
    }, "provenance")
    if provenance["class"] != "fixture_verified":
        raise RuntimeContractError("fixture dispatch requires fixture-verified provenance")
    _digest(provenance["fixture_manifest_digest"], "fixture manifest digest")
    logging = _mapping(document["logging"], {"external_reporting", "structured_metrics_steps"}, "logging")
    if logging["external_reporting"] != "none":
        raise RuntimeContractError("external reporting is prohibited")
    _positive_int(logging["structured_metrics_steps"], "logging steps")
    checkpoints = _mapping(document["checkpoints"], {"limit", "resume", "steps", "strategy"}, "checkpoints")
    if checkpoints["resume"] is not None or checkpoints["strategy"] not in {"none", "steps"}:
        raise RuntimeContractError("checkpoint contract is unsupported")

    evidence = bound.runtime_binding_evidence
    if type(evidence) is not FixtureRuntimeBindingEvidenceV1:
        raise RuntimeContractError("decoder returned the wrong nominal evidence class")
    base_expected = {
        "dataset_revision": dataset["revision"],
        "dependency_lock_digest": runtime["dependency_lock_digest"],
        "engine_commit": code["engine_commit"],
        "model_revision": model["revision"],
        "source_digest": code["source_digest"],
        "target_modules": targets,
        "target_profile_digest": adapter["target_profile_digest"],
        "target_profile_id": adapter["target_profile_id"],
        "tokenizer_revision": model["tokenizer_revision"],
        "workload_digest": digest,
    }
    for name, expected_value in base_expected.items():
        if getattr(evidence, name) != expected_value:
            raise RuntimeContractError("runtime capability is not bound to the workload")
    if evidence.fixture_manifest_digest != provenance["fixture_manifest_digest"]:
        raise RuntimeContractError("fixture evidence does not match provenance")
    return bound, document
