"""Immutable model, tokenizer, dataset, and target-profile loading."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .contracts import HubAccessV1, RuntimeContractError


def _snapshot(repository: str, revision: str, repository_type: str, access: HubAccessV1) -> Path:
    from huggingface_hub import snapshot_download

    token = access.token_for(repository, repository_type)
    snapshot = Path(snapshot_download(
        repo_id=repository,
        repo_type=None if repository_type == "model" else repository_type,
        revision=revision,
        token=token,
        ignore_patterns=["*.bin", "*.pt", "*.pth"] if repository_type == "model" else None,
        local_files_only=True,
    )).resolve()
    if snapshot.name != revision:
        raise RuntimeContractError("resolved snapshot does not match immutable revision")
    return snapshot


def load_model_v1(model_spec: Mapping[str, Any], objective: Mapping[str, Any], access: HubAccessV1):
    snapshot = _snapshot(model_spec["repository"], model_spec["revision"], "model", access)
    if not any(snapshot.rglob("*.safetensors")):
        raise RuntimeContractError("model snapshot contains no safetensors weights")
    import torch
    from unsloth import FastLanguageModel

    model, _ = FastLanguageModel.from_pretrained(
        model_name=str(snapshot),
        max_seq_length=objective["max_seq_length"],
        dtype=torch.bfloat16,
        load_in_4bit=False,
        token=access.token_for(model_spec["repository"], "model"),
        use_exact_model_name=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    source = getattr(getattr(model, "config", None), "_name_or_path", None)
    if not isinstance(source, str) or Path(source).resolve() != snapshot:
        raise RuntimeContractError("loaded model does not match resolved snapshot")
    return model, {
        "repository": model_spec["repository"],
        "requested_revision": model_spec["revision"],
        "loaded_revision": snapshot.name,
        "weights_format": "safetensors",
        "trust_remote_code": False,
    }


def load_tokenizer_v1(model_spec: Mapping[str, Any], access: HubAccessV1):
    snapshot = _snapshot(
        model_spec["repository"], model_spec["tokenizer_revision"], "model", access
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), trust_remote_code=False, local_files_only=True
    )
    source = getattr(tokenizer, "name_or_path", None)
    if not isinstance(source, str) or Path(source).resolve() != snapshot:
        raise RuntimeContractError("loaded tokenizer does not match resolved snapshot")
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template.strip():
        raise RuntimeContractError("tokenizer embedded chat template is required")
    return tokenizer, {
        "repository": model_spec["repository"],
        "requested_revision": model_spec["tokenizer_revision"],
        "loaded_revision": snapshot.name,
        "trust_remote_code": False,
    }


def load_remote_dataset_v1(dataset_spec: Mapping[str, Any], access: HubAccessV1):
    from huggingface_hub import hf_hub_download
    from datasets import load_dataset

    repository = dataset_spec["repository"]
    revision = dataset_spec["revision"]
    selector = dataset_spec["file_selector"]
    split = dataset_spec["split"]
    resolved = Path(hf_hub_download(
        repo_id=repository,
        repo_type="dataset",
        filename=selector,
        revision=revision,
        token=access.token_for(repository, "dataset"),
        local_files_only=True,
    )).resolve()
    normalized = resolved.as_posix()
    if f"/snapshots/{revision}/" not in normalized or not normalized.endswith(f"/{selector}"):
        raise RuntimeContractError("resolved dataset does not match revision and selector")
    dataset = load_dataset("json", data_files={split: str(resolved)}, split=split)
    rows = tuple(dict(row) for row in dataset)
    if not rows:
        raise RuntimeContractError("dataset is empty")
    return rows, {
        "repository": repository,
        "requested_revision": revision,
        "loaded_revision": revision,
        "file_selector": selector,
        "split": split,
        "row_count": len(rows),
    }


def apply_adapter_v1(model: Any, adapter: Mapping[str, Any], optimization: Mapping[str, Any]):
    targets = tuple(adapter["target_modules"])
    names = tuple(name for name, _ in model.named_modules())
    matches: list[dict[str, Any]] = []
    for target in targets:
        found = tuple(name for name in names if name == target or name.endswith(f".{target}"))
        if not found:
            raise RuntimeContractError(f"model does not match target profile member {target}")
        matches.append({"target": target, "matches": list(found)})
    from unsloth import FastLanguageModel

    adapted = FastLanguageModel.get_peft_model(
        model,
        r=adapter["rank"],
        target_modules=list(targets),
        lora_alpha=adapter["alpha"],
        lora_dropout=float(adapter["dropout"]),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=optimization["seed"],
        use_rslora=False,
        use_dora=False,
    )
    return adapted, {
        "target_profile_id": adapter["target_profile_id"],
        "target_profile_digest": adapter["target_profile_digest"],
        "target_modules": list(targets),
        "matches": matches,
    }
