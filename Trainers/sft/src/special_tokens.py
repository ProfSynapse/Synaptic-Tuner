"""Config-driven special-token preparation for SFT.

The module deliberately knows nothing about token semantics.  It transports an
ordered list from configuration, resizes and initializes the model before LoRA,
and describes the exact rows that PEFT must expose through its selective
``trainable_token_indices`` adapter.
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from torch import nn


_EXISTING_TOKEN_POLICIES = {"error", "reuse"}
_INITIALIZATION_POLICIES = {"mean_existing_rows"}
_MERGED_MODEL_SAVE_METHODS = {"merged_16bit", "merged_4bit_forced"}
_SUPPORTED_BNB_4BIT_QUANT_TYPES = {"fp4", "nf4"}
_FULL_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _config_dict(config: Any) -> Dict[str, Any]:
    return {
        "additional_special_tokens": list(config.additional_special_tokens),
        "existing_token_policy": config.existing_token_policy,
        "initialization": config.initialization,
        "train_new_embedding_rows": bool(config.train_new_embedding_rows),
        "train_new_lm_head_rows": bool(config.train_new_lm_head_rows),
        "verify_tokenizer_roundtrip": bool(config.verify_tokenizer_roundtrip),
        "verify_adapter_roundtrip": bool(config.verify_adapter_roundtrip),
        "verify_merged_model_roundtrip": bool(
            getattr(config, "verify_merged_model_roundtrip", False)
        ),
        "merged_model_save_method": getattr(
            config, "merged_model_save_method", "merged_16bit"
        ),
    }


def validate_special_token_config(config: Any) -> None:
    """Validate the generic tokenizer block before model mutation."""
    tokens = config.additional_special_tokens
    if not isinstance(tokens, list):
        raise ValueError("model.tokenizer.additional_special_tokens must be a YAML list.")
    for index, token in enumerate(tokens):
        if not isinstance(token, str) or not token.strip():
            raise ValueError(
                "model.tokenizer.additional_special_tokens entries must be non-empty strings; "
                f"entry {index} is {token!r}."
            )
    if len(tokens) != len(set(tokens)):
        duplicates = sorted({token for token in tokens if tokens.count(token) > 1})
        raise ValueError(
            "model.tokenizer.additional_special_tokens contains duplicates: "
            + ", ".join(repr(token) for token in duplicates)
        )
    if config.existing_token_policy not in _EXISTING_TOKEN_POLICIES:
        raise ValueError(
            "model.tokenizer.existing_token_policy must be one of "
            f"{sorted(_EXISTING_TOKEN_POLICIES)}, got {config.existing_token_policy!r}."
        )
    if config.initialization not in _INITIALIZATION_POLICIES:
        raise ValueError(
            "model.tokenizer.initialization must be one of "
            f"{sorted(_INITIALIZATION_POLICIES)}, got {config.initialization!r}."
        )
    for field_name in (
        "train_new_embedding_rows",
        "train_new_lm_head_rows",
        "verify_tokenizer_roundtrip",
        "verify_adapter_roundtrip",
        "verify_merged_model_roundtrip",
    ):
        value = getattr(config, field_name, False)
        if type(value) is not bool:
            raise ValueError(f"model.tokenizer.{field_name} must be a YAML boolean, got {value!r}.")
    save_method = getattr(config, "merged_model_save_method", "merged_16bit")
    if not isinstance(save_method, str) or save_method not in _MERGED_MODEL_SAVE_METHODS:
        raise ValueError(
            "model.tokenizer.merged_model_save_method must be one of "
            f"{sorted(_MERGED_MODEL_SAVE_METHODS)}, got {save_method!r}."
        )


def _require_tokenizer_api(tokenizer: Any) -> None:
    required = (
        "get_vocab",
        "add_special_tokens",
        "convert_tokens_to_ids",
        "encode",
        "save_pretrained",
    )
    missing = [name for name in required if not callable(getattr(tokenizer, name, None))]
    if missing:
        raise TypeError(
            "Configured special tokens require a text tokenizer with methods: "
            + ", ".join(required)
            + f". Missing: {', '.join(missing)}."
        )
    if not hasattr(tokenizer, "all_special_tokens"):
        raise TypeError(
            "Configured special tokens require tokenizer.all_special_tokens so existing registrations can be verified."
        )


def _registered_special_token_ids(tokenizer: Any) -> Dict[str, int]:
    tokens = getattr(tokenizer, "all_special_tokens", ())
    return {token: int(tokenizer.convert_tokens_to_ids(token)) for token in tokens}


def _add_special_tokens_without_replacement(tokenizer: Any, tokens: list[str]) -> int:
    """Use the tokenizer's declared safe no-replacement API.

    Transformers renamed the keyword without changing the operation.  Detect
    the callable surface itself rather than branching on package versions, and
    prove that every pre-existing special registration survived unchanged.
    """
    add_special_tokens = tokenizer.add_special_tokens
    try:
        parameters = inspect.signature(add_special_tokens).parameters
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Could not inspect tokenizer.add_special_tokens; refusing to risk replacing existing special tokens."
        ) from exc
    if "replace_extra_special_tokens" in parameters:
        no_replace_keyword = "replace_extra_special_tokens"
    elif "replace_additional_special_tokens" in parameters:
        no_replace_keyword = "replace_additional_special_tokens"
    else:
        raise TypeError(
            "tokenizer.add_special_tokens exposes neither replace_extra_special_tokens nor "
            "replace_additional_special_tokens; no safe no-replacement registration surface is available."
        )

    before_specials = _registered_special_token_ids(tokenizer)
    added_count = add_special_tokens(
        {"additional_special_tokens": tokens},
        **{no_replace_keyword: False},
    )
    after_specials = _registered_special_token_ids(tokenizer)
    changed = {
        token: (token_id, after_specials.get(token))
        for token, token_id in before_specials.items()
        if after_specials.get(token) != token_id
    }
    if changed:
        raise RuntimeError(
            "Tokenizer changed or removed pre-existing special-token registrations despite no-replacement mode: "
            f"{changed!r}."
        )
    missing_new = [token for token in tokens if token not in after_specials]
    if missing_new:
        raise RuntimeError(
            "Tokenizer added configured strings but did not register them as special tokens: "
            + ", ".join(repr(token) for token in missing_new)
        )
    return int(added_count)


def _module_name(model: nn.Module, target: nn.Module, description: str) -> str:
    names = [name for name, module in model.named_modules() if module is target and name]
    if not names:
        raise ValueError(f"Could not resolve the model module name for {description}.")
    return min(names, key=lambda name: (name.count("."), len(name)))


def _weight_rows(module: nn.Module, description: str) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise TypeError(f"{description} must expose a two-dimensional .weight tensor.")
    return weight


def _are_weights_tied(input_module: nn.Module, output_module: Optional[nn.Module]) -> bool:
    if output_module is None:
        return False
    input_weight = _weight_rows(input_module, "input embedding")
    output_weight = _weight_rows(output_module, "output head")
    return input_weight is output_weight or input_weight.data_ptr() == output_weight.data_ptr()


def _initialize_rows(weight: torch.Tensor, token_ids: Iterable[int], source_rows: int) -> None:
    token_ids = list(token_ids)
    if not token_ids:
        return
    if source_rows <= 0 or source_rows > weight.shape[0]:
        raise ValueError(
            f"Cannot initialize token rows from source_rows={source_rows} for weight shape {tuple(weight.shape)}."
        )
    with torch.no_grad():
        mean_row = weight[:source_rows].to(dtype=torch.float32).mean(dim=0).to(dtype=weight.dtype)
        index = torch.tensor(token_ids, device=weight.device, dtype=torch.long)
        weight.index_copy_(0, index, mean_row.unsqueeze(0).expand(len(token_ids), -1))


def prepare_special_tokens(model: nn.Module, tokenizer: Any, config: Any) -> Optional[Dict[str, Any]]:
    """Add configured tokens and return the resolved PEFT/lineage contract.

    The returned ``trainable_token_indices`` mapping is passed to LoRA creation.
    Only newly added rows appear in it; collision-policy ``reuse`` never turns a
    pre-existing row into a trainable parameter.
    """
    validate_special_token_config(config)
    tokens = list(config.additional_special_tokens)
    if not tokens:
        return None

    _require_tokenizer_api(tokenizer)
    before_vocab = dict(tokenizer.get_vocab())
    before_tokenizer_size = len(tokenizer)
    collisions = [token for token in tokens if token in before_vocab]
    if collisions and config.existing_token_policy == "error":
        raise ValueError(
            "Configured special tokens already exist in the tokenizer vocabulary: "
            + ", ".join(repr(token) for token in collisions)
            + ". Set model.tokenizer.existing_token_policy=reuse to accept them explicitly."
        )

    input_module = model.get_input_embeddings()
    output_module = model.get_output_embeddings()
    if input_module is None:
        raise TypeError("Configured special tokens require model.get_input_embeddings().")
    input_weight_before = _weight_rows(input_module, "input embedding")
    before_model_vocab_size = int(input_weight_before.shape[0])
    output_weight_before = (
        _weight_rows(output_module, "output head") if output_module is not None else None
    )
    if output_weight_before is not None and output_weight_before.shape[0] != before_model_vocab_size:
        raise ValueError(
            "Input embedding and output head vocabulary dimensions differ before token setup: "
            f"{before_model_vocab_size} vs {output_weight_before.shape[0]}."
        )

    tied = _are_weights_tied(input_module, output_module)
    declared_tied = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False))
    if tied and not declared_tied:
        raise ValueError(
            "Input/output weights share storage but model.config.tie_word_embeddings is false; "
            "the runtime cannot safely install a tied selective-row adapter."
        )
    if declared_tied and not tied:
        raise ValueError(
            "model.config.tie_word_embeddings is true but input/output weights do not share storage; "
            "the runtime cannot safely install a tied selective-row adapter."
        )
    if tied and config.train_new_embedding_rows != config.train_new_lm_head_rows:
        raise ValueError(
            "Input embeddings and output head are tied, so train_new_embedding_rows and "
            "train_new_lm_head_rows must have the same value."
        )
    if config.train_new_lm_head_rows and output_module is None:
        raise TypeError(
            "train_new_lm_head_rows=true requires model.get_output_embeddings() to return a module."
        )

    added_count = _add_special_tokens_without_replacement(tokenizer, tokens)
    after_vocab = dict(tokenizer.get_vocab())
    token_ids = [int(tokenizer.convert_tokens_to_ids(token)) for token in tokens]
    newly_added_tokens = [token for token in tokens if token not in before_vocab]
    newly_added_ids = [int(tokenizer.convert_tokens_to_ids(token)) for token in newly_added_tokens]
    if added_count != len(newly_added_tokens):
        raise RuntimeError(
            "Tokenizer reported an unexpected number of added tokens: "
            f"reported={added_count}, expected={len(newly_added_tokens)}."
        )
    if len(set(token_ids)) != len(token_ids):
        raise RuntimeError("Configured special tokens did not resolve to distinct token IDs.")

    required_vocab_size = max(before_model_vocab_size, len(tokenizer), max(token_ids) + 1)
    resize_applied = required_vocab_size != before_model_vocab_size
    if resize_applied:
        model.resize_token_embeddings(required_vocab_size)

    input_module = model.get_input_embeddings()
    output_module = model.get_output_embeddings()
    input_weight = _weight_rows(input_module, "resized input embedding")
    output_weight = _weight_rows(output_module, "resized output head") if output_module is not None else None
    if input_weight.shape[0] < required_vocab_size:
        raise RuntimeError("resize_token_embeddings did not produce the required input vocabulary size.")
    if output_weight is not None and output_weight.shape[0] < required_vocab_size:
        raise RuntimeError("resize_token_embeddings did not produce the required output vocabulary size.")

    if config.initialization == "mean_existing_rows":
        source_rows = min(before_tokenizer_size, before_model_vocab_size)
        _initialize_rows(input_weight, newly_added_ids, source_rows)
        if output_weight is not None and not _are_weights_tied(input_module, output_module):
            _initialize_rows(output_weight, newly_added_ids, source_rows)

    input_name = _module_name(model, input_module, "input embedding")
    output_name = _module_name(model, output_module, "output head") if output_module is not None else None
    trainable_indices: Dict[str, list[int]] = {}
    if newly_added_ids and config.train_new_embedding_rows:
        if not isinstance(input_module, nn.Embedding):
            raise TypeError(
                "Selective input-row training currently requires torch.nn.Embedding (or a subclass); "
                f"got {type(input_module).__name__}."
            )
        trainable_indices[input_name] = list(newly_added_ids)
    if newly_added_ids and config.train_new_lm_head_rows and not tied:
        if not isinstance(output_module, nn.Linear):
            raise TypeError(
                "Selective output-row training currently requires torch.nn.Linear (or a subclass); "
                f"got {type(output_module).__name__}."
            )
        trainable_indices[output_name] = list(newly_added_ids)

    entries = []
    collision_set = set(collisions)
    for token, token_id in zip(tokens, token_ids):
        encoded = list(tokenizer.encode(token, add_special_tokens=False))
        if encoded != [token_id]:
            raise RuntimeError(
                f"Configured special token {token!r} is not atomic after registration: {encoded!r}."
            )
        entries.append(
            {
                "token": token,
                "token_id": token_id,
                "status": "reused" if token in collision_set else "added",
            }
        )

    resolved_config = _config_dict(config)
    return {
        "schema_version": 1,
        "configured_tokens": entries,
        "new_token_ids": newly_added_ids,
        "trainable_token_indices": trainable_indices,
        "input_embedding_module": input_name,
        "output_head_module": output_name,
        "weights_tied": tied,
        "tokenizer_vocab_size_before": before_tokenizer_size,
        "tokenizer_vocab_size_after": len(tokenizer),
        "model_vocab_size_before": before_model_vocab_size,
        "model_vocab_size_after": int(input_weight.shape[0]),
        "resize_applied": resize_applied,
        "resolved_config": resolved_config,
        "config_sha256": _canonical_hash(resolved_config),
        "vocab_sha256_before": _canonical_hash(before_vocab),
        "vocab_sha256_after": _canonical_hash(after_vocab),
    }


def require_peft_trainable_token_support(trainable_token_indices: Dict[str, list[int]]) -> None:
    """Fail before LoRA if the runtime cannot honor row-selective training."""
    if not trainable_token_indices:
        return
    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise RuntimeError(
            "Selective special-token row training requires PEFT with trainable_token_indices support."
        ) from exc
    if "trainable_token_indices" not in inspect.signature(LoraConfig).parameters:
        raise RuntimeError(
            "This PEFT version does not support LoraConfig.trainable_token_indices. "
            "Use a current PEFT/Unsloth runtime or disable both row-training flags explicitly."
        )


def _find_token_wrapper(named_modules: Dict[str, nn.Module], expected_name: str) -> nn.Module:
    matches = [
        module
        for name, module in named_modules.items()
        if name == expected_name or name.endswith("." + expected_name)
    ]
    wrappers = [module for module in matches if hasattr(module, "token_adapter")]
    if len(wrappers) != 1:
        raise RuntimeError(
            "LoRA did not install exactly one selective token adapter for "
            f"{expected_name!r}; found {len(wrappers)}."
        )
    return wrappers[0]


def _verify_wrapper_rows(
    wrapper: nn.Module,
    expected_name: str,
    expected_ids: list[int],
    *,
    require_trainable: bool,
) -> str:
    adapter_names = list(wrapper.token_adapter.token_indices)
    if len(adapter_names) != 1:
        raise RuntimeError(
            f"Selective token adapter for {expected_name!r} has {len(adapter_names)} active definitions."
        )
    adapter_name = adapter_names[0]
    actual_ids = list(wrapper.token_adapter.token_indices[adapter_name])
    if actual_ids != list(expected_ids):
        raise RuntimeError(
            f"Selective token adapter for {expected_name!r} targets {actual_ids}, expected {expected_ids}."
        )
    delta = wrapper.token_adapter.trainable_tokens_delta[adapter_name]
    if delta.shape[0] != len(expected_ids):
        raise RuntimeError(
            f"Selective token adapter for {expected_name!r} does not expose exactly the requested trainable rows."
        )
    if require_trainable and not delta.requires_grad:
        raise RuntimeError(
            f"Selective token adapter delta for {expected_name!r} is frozen after model preparation."
        )
    if wrapper.token_adapter.get_base_layer().weight.requires_grad:
        raise RuntimeError(
            f"Base vocabulary weight for {expected_name!r} is trainable; old rows would not remain frozen."
        )
    return adapter_name


def verify_peft_trainable_token_adapters(
    model: nn.Module, metadata: Dict[str, Any], *, require_trainable: bool = True
) -> None:
    """Prove LoRA installed the resolved selective-row contract.

    For tied vocabularies PEFT receives only the input embedding in
    ``trainable_token_indices`` and is responsible for installing a second
    output wrapper that points back to the same token adapter.  Verify that
    relation explicitly so a missing or independently-trainable output head can
    never pass merely because the input wrapper exists.
    """
    trainable_token_indices = metadata.get("trainable_token_indices", {})
    if not trainable_token_indices:
        return
    named_modules = dict(model.named_modules())
    verified_wrappers: Dict[str, nn.Module] = {}
    for expected_name, expected_ids in trainable_token_indices.items():
        wrapper = _find_token_wrapper(named_modules, expected_name)
        _verify_wrapper_rows(
            wrapper, expected_name, expected_ids, require_trainable=require_trainable
        )
        verified_wrappers[expected_name] = wrapper

    if not metadata.get("weights_tied", False):
        return
    resolved_config = metadata.get("resolved_config", {})
    tied_row_training = bool(
        metadata.get("new_token_ids")
        and resolved_config.get("train_new_embedding_rows")
        and resolved_config.get("train_new_lm_head_rows")
    )
    if not tied_row_training:
        return

    input_name = metadata.get("input_embedding_module")
    output_name = metadata.get("output_head_module")
    expected_ids = list(metadata.get("new_token_ids", []))
    if not input_name or not output_name:
        raise RuntimeError("Tied selective-row verification requires resolved input and output module names.")
    input_wrapper = verified_wrappers.get(input_name) or _find_token_wrapper(named_modules, input_name)
    input_adapter_name = _verify_wrapper_rows(
        input_wrapper, input_name, expected_ids, require_trainable=require_trainable
    )
    output_wrapper = _find_token_wrapper(named_modules, output_name)
    output_adapter_name = _verify_wrapper_rows(
        output_wrapper, output_name, expected_ids, require_trainable=require_trainable
    )
    if output_adapter_name != input_adapter_name:
        raise RuntimeError(
            "Tied output token adapter uses a different active adapter name from the input embedding."
        )
    input_adapter = input_wrapper.token_adapter
    output_adapter = output_wrapper.token_adapter
    if output_adapter.tied_adapter is not input_adapter:
        raise RuntimeError(
            "Tied output token adapter does not point to the exact input token adapter."
        )
    if (
        output_adapter.trainable_tokens_delta is not input_adapter.trainable_tokens_delta
        or output_adapter.trainable_tokens_original is not input_adapter.trainable_tokens_original
        or output_adapter.token_indices is not input_adapter.token_indices
    ):
        raise RuntimeError(
            "Tied output token adapter exposes independent row state instead of sharing the input adapter."
        )
    if (
        output_adapter.trainable_tokens_delta[output_adapter_name]
        is not input_adapter.trainable_tokens_delta[input_adapter_name]
    ):
        raise RuntimeError("Tied output token adapter exposes an independent duplicate delta parameter.")


def restore_verified_selective_token_deltas(
    model: nn.Module, metadata: Dict[str, Any]
) -> tuple[nn.Parameter, ...]:
    """Restore only structurally verified PEFT row deltas after runtime patching.

    Some optimized model-preparation paths freeze every parameter whose name is
    not LoRA-shaped.  Selective-token deltas are PEFT adapter parameters but do
    not use a LoRA name.  First verify the complete adapter structure while
    permitting frozen deltas, then re-enable the exact resolved Parameter
    objects in float32, and finally rerun strict verification.
    """
    trainable_token_indices = metadata.get("trainable_token_indices", {})
    if not trainable_token_indices:
        return ()
    verify_peft_trainable_token_adapters(model, metadata, require_trainable=False)
    named_modules = dict(model.named_modules())
    deltas: list[nn.Parameter] = []
    seen: set[int] = set()
    for module_name, expected_ids in trainable_token_indices.items():
        wrapper = _find_token_wrapper(named_modules, module_name)
        adapter_name = _verify_wrapper_rows(
            wrapper, module_name, expected_ids, require_trainable=False
        )
        delta = wrapper.token_adapter.trainable_tokens_delta[adapter_name]
        if id(delta) in seen:
            continue
        seen.add(id(delta))
        delta.data = delta.data.to(dtype=torch.float32)
        delta.requires_grad_(True)
        deltas.append(delta)
    verify_peft_trainable_token_adapters(model, metadata, require_trainable=True)
    return tuple(deltas)


def _verify_atomic_tokens(tokenizer: Any, metadata: Dict[str, Any]) -> None:
    for entry in metadata["configured_tokens"]:
        token = entry["token"]
        expected_id = entry["token_id"]
        actual_id = int(tokenizer.convert_tokens_to_ids(token))
        encoded = list(tokenizer.encode(token, add_special_tokens=False))
        if actual_id != expected_id or encoded != [expected_id]:
            raise RuntimeError(
                f"Saved tokenizer round-trip changed special token {token!r}: "
                f"id={actual_id}, encoded={encoded!r}, expected={expected_id}."
            )


def save_special_token_artifacts(
    tokenizer: Any, output_dir: Path | str, metadata: Optional[Dict[str, Any]]
) -> None:
    """Save tokenizer + resolved contract and optionally reload-verify them."""
    if metadata is None:
        return
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output_path))
    verify_saved_special_tokenizer(tokenizer, output_path, metadata)
    write_special_token_lineage(output_path, metadata)


def verify_saved_special_tokenizer(
    tokenizer: Any, output_dir: Path | str, metadata: Optional[Dict[str, Any]]
) -> None:
    """Save the tokenizer and, when configured, verify its local reload."""
    if metadata is None:
        return
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output_path))
    if metadata["resolved_config"]["verify_tokenizer_roundtrip"]:
        loader = getattr(tokenizer.__class__, "from_pretrained", None)
        if not callable(loader):
            raise TypeError(
                "verify_tokenizer_roundtrip=true requires tokenizer.__class__.from_pretrained()."
            )
        reloaded = loader(str(output_path), local_files_only=True)
        _verify_atomic_tokens(reloaded, metadata)
        reloaded_vocab_hash = _canonical_hash(dict(reloaded.get_vocab()))
        if reloaded_vocab_hash != metadata["vocab_sha256_after"]:
            raise RuntimeError(
                "Saved tokenizer vocabulary hash changed on reload: "
                f"{reloaded_vocab_hash} != {metadata['vocab_sha256_after']}."
            )


def write_special_token_lineage(
    output_dir: Path | str, metadata: Optional[Dict[str, Any]]
) -> None:
    """Write the final resolved metadata after all requested checks pass."""
    if metadata is None:
        return
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "special_tokens_lineage.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _selected_vocab_rows(module: nn.Module, token_ids: list[int]) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError("Merged-model verification requires vocabulary modules with tensor weights.")
    return weight.detach()[token_ids].to(device="cpu").contiguous()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _canonical_quantization_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _canonical_quantization_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_quantization_value(item) for item in value]
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)) or enum_value is None:
        if enum_value is not None:
            return enum_value
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "Quantization configuration contains a non-canonical value of type "
        f"{type(value).__name__}."
    )


def _require_bnb_4bit_invariants(
    quantization_config: Any,
    *,
    description: str,
    expected: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if hasattr(quantization_config, "to_dict"):
        quantization_config = quantization_config.to_dict()
    if not isinstance(quantization_config, dict):
        raise RuntimeError(f"{description} does not contain a BitsAndBytes configuration.")
    canonical = _canonical_quantization_value(quantization_config)
    quant_type = canonical.get("bnb_4bit_quant_type")
    if (
        canonical.get("quant_method") != "bitsandbytes"
        or canonical.get("load_in_4bit") is not True
        or not isinstance(quant_type, str)
        or quant_type not in _SUPPORTED_BNB_4BIT_QUANT_TYPES
    ):
        raise RuntimeError(
            f"{description} is not a supported BitsAndBytes 4-bit representation: "
            f"{canonical!r}."
        )
    if expected is not None and canonical != expected:
        raise RuntimeError(
            f"{description} canonical BitsAndBytes config differs from expected: "
            f"{canonical!r} != {expected!r}."
        )
    return canonical


def _fresh_forced_4bit_merge_candidate(
    output_dir: Path, metadata: Dict[str, Any]
) -> tuple[nn.Module, str, Dict[str, Any], Path]:
    """Load a separate pinned base and attach the saved adapter for destructive merge."""
    provenance = metadata["base_model_provenance"]
    repo = provenance["requested_repo"]
    revision = provenance["requested_revision"]
    from huggingface_hub import snapshot_download

    snapshot_path = Path(
        snapshot_download(
            repo_id=repo,
            revision=revision,
            local_files_only=True,
        )
    ).resolve()
    if not snapshot_path.is_dir() or snapshot_path.name.lower() != revision.lower():
        raise RuntimeError(
            "Fresh forced-merge base did not resolve to the exact pinned local snapshot: "
            f"{snapshot_path}."
        )
    device_map: Dict[str, Any]
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        device_map = {"": device_index}
        device_label = f"cuda:{device_index}"
    else:
        device_map = {"": "cpu"}
        device_label = "cpu"

    from transformers import AutoModelForCausalLM

    base_model = AutoModelForCausalLM.from_pretrained(
        repo,
        revision=revision,
        local_files_only=True,
        trust_remote_code=False,
        device_map=device_map,
        dtype="auto",
        low_cpu_mem_usage=True,
    )
    invariants = _require_bnb_4bit_invariants(
        getattr(base_model.config, "quantization_config", None),
        description="Fresh forced-merge base",
    )
    if getattr(base_model, "is_loaded_in_4bit", False) is not True:
        raise RuntimeError("Fresh forced-merge base was not loaded in 4-bit mode.")

    expected_vocab_size = int(metadata["model_vocab_size_after"])
    actual_vocab_size = int(base_model.get_input_embeddings().weight.shape[0])
    if actual_vocab_size > expected_vocab_size:
        raise RuntimeError(
            "Fresh pinned base vocabulary is larger than the recorded prepared model: "
            f"{actual_vocab_size} > {expected_vocab_size}."
        )
    if actual_vocab_size < expected_vocab_size:
        base_model.resize_token_embeddings(expected_vocab_size)

    from peft import PeftModel

    candidate = PeftModel.from_pretrained(
        base_model,
        str(output_dir),
        is_trainable=False,
        local_files_only=True,
    )
    verify_peft_trainable_token_adapters(candidate, metadata, require_trainable=False)

    from unsloth.save import patch_saving_functions

    candidate = patch_saving_functions(candidate)
    if not callable(getattr(candidate, "save_pretrained_merged", None)):
        raise TypeError("Unsloth did not install save_pretrained_merged on the merge copy.")
    return candidate, device_label, invariants, snapshot_path


def _module_topology_signature(model: nn.Module) -> list[tuple[str, int]]:
    return [(name, id(module)) for name, module in model.named_modules()]


def _validate_unsloth_skip_module_mutation(
    source: Dict[str, Any], mutated: Dict[str, Any]
) -> tuple[list[str], Dict[str, Any]]:
    source_other = dict(source)
    mutated_other = dict(mutated)
    source_other.pop("llm_int8_skip_modules", None)
    mutated_other.pop("llm_int8_skip_modules", None)
    if mutated_other != source_other:
        raise RuntimeError(
            "Unsloth forced save changed canonical quantization fields other than "
            "llm_int8_skip_modules."
        )
    skip_modules = mutated.get("llm_int8_skip_modules")
    if skip_modules is None:
        computed: list[str] = []
    elif (
        not isinstance(skip_modules, list)
        or any(not isinstance(name, str) or not name for name in skip_modules)
        or len(skip_modules) != len(set(skip_modules))
    ):
        raise RuntimeError(
            "Unsloth computed llm_int8_skip_modules must be a unique list of non-empty strings."
        )
    else:
        computed = list(skip_modules)
    return computed, mutated


def _compute_post_merge_skip_modules(model: nn.Module) -> list[str]:
    get_base_model = getattr(model, "get_base_model", None)
    if not callable(get_base_model):
        raise TypeError("Forced-merge candidate cannot expose its post-merge base model.")
    post_merge_base = get_base_model()
    from unsloth_zoo.saving_utils import find_skipped_quantized_modules

    skipped, _ = find_skipped_quantized_modules(post_merge_base)
    if (
        not isinstance(skipped, list)
        or any(not isinstance(name, str) or not name for name in skipped)
        or len(skipped) != len(set(skipped))
    ):
        raise RuntimeError("Post-merge skipped modules are not a canonical name list.")
    return list(skipped)


def _save_merged_with_quantization_config_compat(
    model: nn.Module,
    save_merged: Any,
    save_directory: Path,
    tokenizer: Any,
    *,
    save_method: str,
    pinned_local_snapshot: Optional[Path] = None,
    source_quantization: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Bridge one Unsloth forced-save branch that indexes BnB config as a dict."""
    if save_method != "merged_4bit_forced":
        save_merged(str(save_directory), tokenizer, save_method=save_method)
        return {"applied": False, "reason": "save_method_not_forced_4bit"}

    if pinned_local_snapshot is None:
        raise ValueError("Forced 4-bit merge requires an exact pinned local snapshot path.")
    pinned_local_snapshot = Path(pinned_local_snapshot).resolve()
    if not pinned_local_snapshot.is_dir():
        raise ValueError(
            f"Forced 4-bit pinned local snapshot is missing: {pinned_local_snapshot}."
        )
    model_config = getattr(model, "config", None)
    quantization_config = getattr(model_config, "quantization_config", None)
    if source_quantization is None:
        raise ValueError("Forced 4-bit merge requires captured source quantization config.")
    original_quantization = _require_bnb_4bit_invariants(
        quantization_config,
        description="Forced-save candidate quantization config",
        expected=source_quantization,
    )
    temporary_mapping = _canonical_quantization_value(original_quantization)
    if temporary_mapping is original_quantization:
        raise RuntimeError("Forced-save quantization mapping was not detached.")
    original_name_present = hasattr(model_config, "_name_or_path")
    original_name_or_path = getattr(model_config, "_name_or_path", None)
    model_config.quantization_config = temporary_mapping
    model_config._name_or_path = str(pinned_local_snapshot)
    mutated_quantization = None
    computed_skip_modules = None
    try:
        save_merged(str(save_directory), tokenizer, save_method=save_method)
        mutated_quantization = _require_bnb_4bit_invariants(
            temporary_mapping,
            description="Post-Unsloth forced-save quantization config",
        )
        observed_skip_modules, mutated_quantization = (
            _validate_unsloth_skip_module_mutation(
                original_quantization, mutated_quantization
            )
        )
        computed_skip_modules = _compute_post_merge_skip_modules(model)
        if observed_skip_modules != computed_skip_modules:
            raise RuntimeError(
                "Unsloth llm_int8_skip_modules does not match the post-merge module scan: "
                f"{observed_skip_modules!r} != {computed_skip_modules!r}."
            )
    finally:
        model_config.quantization_config = quantization_config
        if original_name_present:
            model_config._name_or_path = original_name_or_path
        else:
            delattr(model_config, "_name_or_path")
    if model_config.quantization_config is not quantization_config:
        raise RuntimeError("Forced-save compatibility shim did not restore quantization_config.")
    if _require_bnb_4bit_invariants(
        model_config.quantization_config,
        description="Restored forced-save quantization config",
    ) != original_quantization:
        raise RuntimeError("Forced-save compatibility shim changed quantization config values.")
    if original_name_present:
        name_restored = model_config._name_or_path == original_name_or_path
    else:
        name_restored = not hasattr(model_config, "_name_or_path")
    if not name_restored:
        raise RuntimeError("Forced-save compatibility shim did not restore _name_or_path.")
    return {
        "applied": True,
        "reason": "unsloth_detached_mapping_compatibility",
        "original_type": type(quantization_config).__name__,
        "original_mapping_detached": True,
        "computed_skip_modules": computed_skip_modules,
        "expected_saved_quantization": mutated_quantization,
        "pinned_local_snapshot_used": True,
        "restored": True,
    }


def verify_merged_model_roundtrip(
    model: nn.Module,
    tokenizer: Any,
    output_dir: Path | str,
    metadata: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Opt-in smoke check for merge -> save -> fresh local reload.

    The merged model is temporary and never published.  Success leaves only a
    compact report, which is also bound into ``special_tokens_lineage.json``.
    """
    if metadata is None or not metadata["resolved_config"].get(
        "verify_merged_model_roundtrip", False
    ):
        return None
    resolved_config = metadata.get("resolved_config")
    save_method = (
        resolved_config.get("merged_model_save_method")
        if isinstance(resolved_config, dict)
        else None
    )
    if not isinstance(save_method, str) or save_method not in _MERGED_MODEL_SAVE_METHODS:
        raise ValueError(
            "Merged-model verification requires an explicit supported "
            "model.tokenizer.merged_model_save_method; got "
            f"{save_method!r}."
        )
    live_save_merged = getattr(model, "save_pretrained_merged", None)
    if save_method == "merged_16bit" and not callable(live_save_merged):
        raise TypeError(
            "merged_16bit verification requires model.save_pretrained_merged()."
        )
    token_ids = [int(entry["token_id"]) for entry in metadata["configured_tokens"]]
    if not token_ids:
        raise RuntimeError("Merged-model verification was requested without configured tokens.")
    base_provenance = metadata.get("base_model_provenance")
    if (
        not isinstance(base_provenance, dict)
        or not base_provenance.get("portable")
        or not _FULL_COMMIT.fullmatch(base_provenance.get("requested_revision") or "")
        or base_provenance.get("resolved_commit")
        != base_provenance.get("requested_revision")
    ):
        raise RuntimeError(
            "Merged-model verification requires validated original-repo and pinned-revision evidence."
        )

    input_before = _selected_vocab_rows(model.get_input_embeddings(), token_ids)
    output_module = model.get_output_embeddings()
    output_before = (
        _selected_vocab_rows(output_module, token_ids) if output_module is not None else None
    )
    live_topology_before = _module_topology_signature(model)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=".merged-roundtrip-", dir=str(output_path.parent))
    )
    reloaded_model = None
    reloaded_tokenizer = None
    merge_candidate = None
    merge_candidate_device = None
    merge_candidate_snapshot = None
    source_quantization = None
    save_compatibility = None
    report: Optional[Dict[str, Any]] = None
    try:
        if save_method == "merged_4bit_forced":
            (
                merge_candidate,
                merge_candidate_device,
                source_quantization,
                merge_candidate_snapshot,
            ) = (
                _fresh_forced_4bit_merge_candidate(output_path, metadata)
            )
            if merge_candidate is model:
                raise RuntimeError("Forced 4-bit merge candidate must not be the live model.")
            candidate_input = _selected_vocab_rows(
                merge_candidate.get_input_embeddings(), token_ids
            )
            if not torch.equal(input_before.to(dtype=candidate_input.dtype), candidate_input):
                raise RuntimeError(
                    "Fresh forced-merge candidate input rows differ from the live model."
                )
            candidate_output_module = merge_candidate.get_output_embeddings()
            if (output_before is None) != (candidate_output_module is None):
                raise RuntimeError(
                    "Fresh forced-merge candidate changed output vocabulary module presence."
                )
            if output_before is not None:
                candidate_output = _selected_vocab_rows(
                    candidate_output_module, token_ids
                )
                if not torch.equal(
                    output_before.to(dtype=candidate_output.dtype), candidate_output
                ):
                    raise RuntimeError(
                        "Fresh forced-merge candidate output rows differ from the live model."
                    )
            save_merged = merge_candidate.save_pretrained_merged
        else:
            save_merged = live_save_merged

        save_compatibility = _save_merged_with_quantization_config_compat(
            merge_candidate if save_method == "merged_4bit_forced" else model,
            save_merged,
            temporary_dir,
            tokenizer,
            save_method=save_method,
            pinned_local_snapshot=merge_candidate_snapshot,
            source_quantization=source_quantization,
        )
        if _module_topology_signature(model) != live_topology_before:
            raise RuntimeError("Merged-model verification changed the live model topology.")
        live_input_after_save = _selected_vocab_rows(
            model.get_input_embeddings(), token_ids
        )
        if not torch.equal(input_before, live_input_after_save):
            raise RuntimeError(
                "Merged-model verification changed live configured input rows."
            )
        live_output_after_save = model.get_output_embeddings()
        if (output_before is None) != (live_output_after_save is None):
            raise RuntimeError(
                "Merged-model verification changed live output vocabulary module presence."
            )
        if output_before is not None and not torch.equal(
            output_before,
            _selected_vocab_rows(live_output_after_save, token_ids),
        ):
            raise RuntimeError(
                "Merged-model verification changed live configured output rows."
            )
        save_merged = None
        del merge_candidate
        merge_candidate = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        saved_paths = sorted(path for path in temporary_dir.rglob("*") if path.is_file())
        if not saved_paths:
            raise RuntimeError("Merged-model save produced no files.")
        saved_config_path = temporary_dir / "config.json"
        if not saved_config_path.is_file():
            raise RuntimeError("Merged-model save omitted config.json.")
        saved_config = json.loads(saved_config_path.read_text(encoding="utf-8"))
        if save_method == "merged_4bit_forced":
            saved_quantization = _require_bnb_4bit_invariants(
                saved_config.get("quantization_config"),
                description="Saved forced-merge model",
                expected=save_compatibility["expected_saved_quantization"],
            )
            saved_representation = (
                "bitsandbytes_"
                f"{source_quantization['bnb_4bit_quant_type']}_4bit"
            )
        else:
            if saved_config.get("quantization_config") is not None:
                raise RuntimeError(
                    "Saved merged_16bit model unexpectedly retained quantization_config."
                )
            saved_quantization = None
            saved_representation = "unquantized_16bit"

        from transformers import AutoModelForCausalLM, AutoTokenizer

        reloaded_tokenizer = AutoTokenizer.from_pretrained(
            str(temporary_dir), local_files_only=True, trust_remote_code=False
        )
        _verify_atomic_tokens(reloaded_tokenizer, metadata)
        reloaded_model = AutoModelForCausalLM.from_pretrained(
            str(temporary_dir),
            local_files_only=True,
            trust_remote_code=False,
            device_map={"": "cpu"},
            dtype="auto",
            low_cpu_mem_usage=True,
        )
        if save_method == "merged_4bit_forced":
            if getattr(reloaded_model, "is_loaded_in_4bit", False) is not True:
                raise RuntimeError("Fresh forced-merge reload was not loaded in 4-bit mode.")
            _require_bnb_4bit_invariants(
                getattr(reloaded_model.config, "quantization_config", None),
                description="Fresh forced-merge reload",
                expected=saved_quantization,
            )
        elif getattr(reloaded_model, "is_loaded_in_4bit", False):
            raise RuntimeError("Fresh merged_16bit reload unexpectedly loaded in 4-bit mode.")
        input_after = _selected_vocab_rows(reloaded_model.get_input_embeddings(), token_ids)
        if not torch.equal(input_before.to(dtype=input_after.dtype), input_after):
            raise RuntimeError("Merged/reloaded input embedding rows differ for configured tokens.")

        reloaded_output = reloaded_model.get_output_embeddings()
        if (output_before is None) != (reloaded_output is None):
            raise RuntimeError("Merged/reloaded model changed output vocabulary module presence.")
        output_record = None
        if output_before is not None:
            output_after = _selected_vocab_rows(reloaded_output, token_ids)
            if not torch.equal(output_before.to(dtype=output_after.dtype), output_after):
                raise RuntimeError("Merged/reloaded output-head rows differ for configured tokens.")
            output_record = {
                "dtype": str(output_after.dtype).removeprefix("torch."),
                "shape": list(output_after.shape),
                "sha256": _tensor_sha256(output_after),
            }

        report = {
            "schema_version": 2,
            "requested": True,
            "result": "passed",
            "method": f"unsloth_{save_method}_save_fresh_transformers_cpu_reload",
            "save_method": save_method,
            "requested_save_method": save_method,
            "saved_representation": saved_representation,
            "merge_source": (
                "fresh_pinned_base_plus_saved_adapter"
                if save_method == "merged_4bit_forced"
                else "live_model_nondestructive_export"
            ),
            "merge_candidate_device": merge_candidate_device,
            "source_quantization": source_quantization,
            "saved_quantization": saved_quantization,
            "save_compatibility": save_compatibility,
            "reload_loader": "transformers.AutoModelForCausalLM",
            "reload_device": "cpu",
            "local_files_only": True,
            "trust_remote_code": False,
            "reload_is_loaded_in_4bit": bool(
                getattr(reloaded_model, "is_loaded_in_4bit", False)
            ),
            "verification_scope": "serialization_and_configured_token_rows_no_forward",
            "forward_executed": False,
            "live_model_topology_preserved": True,
            "live_model_configured_rows_preserved": True,
            "configured_tokens": [dict(entry) for entry in metadata["configured_tokens"]],
            "base_model_provenance": dict(base_provenance),
            "input_rows": {
                "dtype": str(input_after.dtype).removeprefix("torch."),
                "shape": list(input_after.shape),
                "sha256": _tensor_sha256(input_after),
            },
            "output_rows": output_record,
            "saved_file_manifest": [
                {
                    "path": str(path.relative_to(temporary_dir)),
                    "size_bytes": path.stat().st_size,
                }
                for path in saved_paths
            ],
            "temporary_artifacts_removed": True,
            "published": False,
        }
    finally:
        del merge_candidate
        del reloaded_model
        del reloaded_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        shutil.rmtree(temporary_dir)

    if report is None:
        raise RuntimeError("Merged-model round-trip did not produce a success report.")
    (output_path / "merged_model_roundtrip.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metadata["merged_model_roundtrip"] = report
    return report


def _active_adapter_names(model: nn.Module) -> list[str]:
    active = getattr(model, "active_adapters", None)
    if callable(active):
        active = active()
    if isinstance(active, str):
        return [active]
    if isinstance(active, (list, tuple)):
        return list(active)
    fallback = getattr(model, "active_adapter", None)
    if isinstance(fallback, str):
        return [fallback]
    raise RuntimeError("Cannot determine the live PEFT model's active adapter.")


def _adapter_artifact_state(output_dir: Path | str) -> Dict[str, torch.Tensor]:
    output_path = Path(output_dir)
    safetensors_path = output_path / "adapter_model.safetensors"
    bin_path = output_path / "adapter_model.bin"
    present = [path for path in (safetensors_path, bin_path) if path.is_file()]
    if len(present) != 1:
        raise RuntimeError(
            "Expected exactly one PEFT adapter weight artifact after save; found "
            + ", ".join(path.name for path in present)
        )
    artifact_path = present[0]
    if artifact_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("Reading the saved PEFT adapter requires safetensors.") from exc
        return dict(load_file(str(artifact_path), device="cpu"))
    try:
        return dict(torch.load(artifact_path, map_location="cpu", weights_only=True))
    except TypeError as exc:
        raise RuntimeError(
            "This torch runtime cannot safely load adapter_model.bin with weights_only=True."
        ) from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_manifest(state: Dict[str, torch.Tensor]) -> list[Dict[str, Any]]:
    return [
        {
            "key": key,
            "shape": list(state[key].shape),
            "dtype": str(state[key].dtype).removeprefix("torch."),
        }
        for key in sorted(state)
    ]


def _reject_full_vocab_tensors(
    state: Dict[str, torch.Tensor], metadata: Dict[str, Any]
) -> None:
    module_names = {
        name
        for name in (
            metadata.get("input_embedding_module"),
            metadata.get("output_head_module"),
        )
        if name
    }
    forbidden = []
    for key in state:
        if ".token_adapter.base_layer." in key:
            forbidden.append(key)
            continue
        for module_name in module_names:
            if key == f"{module_name}.weight" or key.endswith(f".{module_name}.weight"):
                forbidden.append(key)
                break
    if forbidden:
        raise RuntimeError(
            "Selective-token adapter artifact contains full base embedding/output tensors: "
            + ", ".join(sorted(forbidden))
        )


def _adapter_only_live_state(model: nn.Module, adapter_name: str) -> Dict[str, torch.Tensor]:
    try:
        from peft import get_peft_model_state_dict
    except ImportError as exc:
        raise RuntimeError("Adapter artifact verification requires PEFT state-dict utilities.") from exc
    return dict(
        get_peft_model_state_dict(
            model,
            adapter_name=adapter_name,
            save_embedding_layers=False,
        )
    )


def _compare_adapter_states(
    expected: Dict[str, torch.Tensor],
    actual: Dict[str, torch.Tensor],
    *,
    actual_description: str,
) -> None:
    if set(expected) != set(actual):
        missing_keys = sorted(set(expected) - set(actual))
        extra_keys = sorted(set(actual) - set(expected))
        raise RuntimeError(
            f"{actual_description} PEFT adapter state keys differ from the live adapter: "
            f"missing={missing_keys}, extra={extra_keys}."
        )
    for key in sorted(expected):
        live_tensor = expected[key].detach()
        actual_tensor = actual[key].detach()
        if live_tensor.shape != actual_tensor.shape or live_tensor.dtype != actual_tensor.dtype:
            raise RuntimeError(
                f"{actual_description} adapter tensor {key!r} changed shape/dtype: "
                f"{tuple(live_tensor.shape)}/{live_tensor.dtype} vs "
                f"{tuple(actual_tensor.shape)}/{actual_tensor.dtype}."
            )
        comparison_tensor = actual_tensor.to(device=live_tensor.device)
        if not torch.equal(live_tensor, comparison_tensor):
            raise RuntimeError(
                f"{actual_description} adapter tensor {key!r} differs from the live saved value."
            )


def restore_portable_adapter_base_provenance(
    model: nn.Module,
    metadata: Optional[Dict[str, Any]],
    *,
    requested_repo: str,
    requested_revision: Optional[str],
    revision_evidence: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Replace runtime cache paths for a pinned Hub revision before adapter save.

    Unpinned runs are a strict no-op so supported local model paths retain their
    existing PEFT save behavior. ``metadata`` is optional because a pinned SFT
    run must retain portable provenance even without additional special tokens.
    """
    merge_requested = bool(
        metadata
        and metadata.get("resolved_config", {}).get(
            "verify_merged_model_roundtrip", False
        )
    )
    if requested_revision is None:
        if merge_requested:
            raise ValueError(
                "verify_merged_model_roundtrip=true requires a full pinned model revision."
            )
        return None

    repo = requested_repo.strip() if isinstance(requested_repo, str) else ""
    if (
        not repo
        or Path(repo).is_absolute()
        or "snapshots/" in repo.replace("\\", "/")
        or repo.startswith((".", "/"))
    ):
        raise ValueError(
            "Portable adapter provenance requires the original Hugging Face repository id, "
            "not a local/cache snapshot path."
        )
    if not _FULL_COMMIT.fullmatch(requested_revision):
        raise ValueError("Portable adapter revision must be a full 40-character commit SHA.")
    if not isinstance(revision_evidence, dict):
        raise RuntimeError("Pinned adapter provenance requires model revision evidence.")
    if (
        revision_evidence.get("requested_repo") != repo
        or revision_evidence.get("requested_revision") != requested_revision
        or revision_evidence.get("resolved_commit") != requested_revision.lower()
    ):
        raise RuntimeError("Adapter base provenance does not match validated revision evidence.")
    resolved_revision = requested_revision.lower()
    resolved_commit = revision_evidence["resolved_commit"]

    peft_configs = getattr(model, "peft_config", None)
    if not isinstance(peft_configs, dict) or not peft_configs:
        raise TypeError("Portable adapter provenance requires a PEFT model with peft_config entries.")
    adapters = []
    for adapter_name, peft_config in sorted(peft_configs.items()):
        peft_config.base_model_name_or_path = repo
        peft_config.revision = resolved_revision
        adapters.append(str(adapter_name))
    record = {
        "requested_repo": repo,
        "requested_revision": resolved_revision,
        "resolved_commit": resolved_commit,
        "adapters": adapters,
        "portable": True,
        "runtime_snapshot_path_persisted": False,
    }
    if metadata is not None:
        metadata["base_model_provenance"] = record
    return record


def verify_saved_adapter_base_provenance(
    output_dir: Path | str,
    provenance: Dict[str, Any],
) -> Dict[str, Any]:
    """Assert every saved PEFT config retains the original repo and revision."""
    if not isinstance(provenance, dict) or not provenance.get("portable"):
        raise RuntimeError("Adapter save requires portable base-model provenance first.")
    expected_repo = provenance.get("requested_repo")
    expected_revision = provenance.get("requested_revision")
    if expected_revision is not None and not _FULL_COMMIT.fullmatch(expected_revision):
        raise RuntimeError("Saved adapter provenance requires a full 40-character revision.")

    output_path = Path(output_dir)
    adapter_config_paths = sorted(output_path.rglob("adapter_config.json"))
    if not adapter_config_paths:
        raise RuntimeError("Saved adapter is missing adapter_config.json.")
    for adapter_config_path in adapter_config_paths:
        try:
            saved_adapter_config = json.loads(
                adapter_config_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Saved adapter config is unreadable: {adapter_config_path}."
            ) from exc
        if saved_adapter_config.get("base_model_name_or_path") != expected_repo:
            raise RuntimeError(
                f"Saved adapter config persisted a non-portable base model path: "
                f"{adapter_config_path}."
            )
        if saved_adapter_config.get("revision") != expected_revision:
            raise RuntimeError(
                f"Saved adapter config persisted the wrong model revision: "
                f"{adapter_config_path}."
            )
    return {
        "requested_repo": expected_repo,
        "requested_revision": expected_revision,
        "adapter_config_paths": [
            str(path.relative_to(output_path)) for path in adapter_config_paths
        ],
        "result": "passed",
    }


def save_adapter_without_base_vocab(
    model: nn.Module,
    output_dir: Path | str,
    metadata: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Rewrite a special-token PEFT artifact without full base vocab tensors.

    PEFT's automatic embedding-save policy includes complete embedding/head
    weights when the tokenizer grew. Selective-token deltas are absolute
    replacement rows, so those full tensors are redundant, large, and unsafe to
    temporary-load into the live model. Re-save explicitly adapter-only and
    prove the on-disk state equals PEFT's adapter-only live state.
    """
    if metadata is None:
        return None
    save_pretrained = getattr(model, "save_pretrained", None)
    if not callable(save_pretrained):
        raise TypeError("Special-token artifact finalization requires PEFT save_pretrained().")
    output_path = Path(output_dir)
    save_pretrained(
        str(output_path),
        safe_serialization=True,
        save_embedding_layers=False,
    )
    provenance = metadata.get("base_model_provenance")
    if provenance is not None:
        verify_saved_adapter_base_provenance(output_path, provenance)
    stale_bin = output_path / "adapter_model.bin"
    if stale_bin.is_file() and (output_path / "adapter_model.safetensors").is_file():
        stale_bin.unlink()
    active_names = _active_adapter_names(model)
    if len(active_names) != 1:
        raise RuntimeError(
            "Adapter-only special-token save requires exactly one active adapter; "
            f"found {active_names!r}."
        )
    live_state = _adapter_only_live_state(model, active_names[0])
    artifact_state = _adapter_artifact_state(output_path)
    _reject_full_vocab_tensors(artifact_state, metadata)
    _compare_adapter_states(live_state, artifact_state, actual_description="On-disk")
    if metadata.get("trainable_token_indices") and not any(
        "trainable_tokens_delta" in key for key in artifact_state
    ):
        raise RuntimeError("Adapter-only artifact omitted configured selective-token deltas.")
    manifest = _tensor_manifest(artifact_state)
    state_file = (
        "adapter_model.safetensors"
        if (output_path / "adapter_model.safetensors").is_file()
        else "adapter_model.bin"
    )
    return {
        "adapter": active_names[0],
        "tensor_count": len(artifact_state),
        "tensor_manifest": manifest,
        "tensor_manifest_sha256": _canonical_hash(manifest),
        "state_file": state_file,
        "save_embedding_layers": False,
        "full_base_vocab_tensors_present": False,
        "save_validation": "passed",
    }


def bind_adapter_artifact_lineage(
    output_dir: Path | str,
    metadata: Dict[str, Any],
    save_report: Dict[str, Any],
    verification_report: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Bind exact adapter bytes and successful checks into resolved metadata."""
    if save_report.get("save_validation") != "passed":
        raise RuntimeError("Cannot record an adapter artifact without a successful save validation.")
    verification_requested = bool(
        metadata["resolved_config"].get("verify_adapter_roundtrip", False)
    )
    if verification_requested and (
        verification_report is None or verification_report.get("result") != "passed"
    ):
        raise RuntimeError("Cannot record adapter artifact success before round-trip verification passes.")
    output_path = Path(output_dir)
    relative_files = ["adapter_config.json", save_report["state_file"]]
    files = []
    for relative_path in relative_files:
        path = output_path / relative_path
        if not path.is_file():
            raise RuntimeError(f"Required adapter artifact file is missing: {relative_path}")
        files.append(
            {
                "path": relative_path,
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    record = {
        "files": files,
        "tensor_count": save_report["tensor_count"],
        "tensor_manifest": save_report["tensor_manifest"],
        "tensor_manifest_sha256": save_report["tensor_manifest_sha256"],
        "save_embedding_layers": False,
        "full_base_vocab_tensors_present": False,
        "save_validation": "passed",
        "verification": verification_report
        if verification_report is not None
        else {
            "method": "disabled_by_config",
            "requested": False,
            "result": "not_requested",
        },
    }
    metadata["adapter_artifact"] = record
    return record


def _verify_loaded_adapter_rows(
    model: nn.Module, metadata: Dict[str, Any], adapter_name: str
) -> None:
    named_modules = dict(model.named_modules())
    trainable_token_indices = metadata.get("trainable_token_indices", {})
    wrappers: Dict[str, nn.Module] = {}
    for module_name, expected_ids in trainable_token_indices.items():
        wrapper = _find_token_wrapper(named_modules, module_name)
        token_adapter = wrapper.token_adapter
        if adapter_name not in token_adapter.token_indices:
            raise RuntimeError(
                f"Reloaded adapter {adapter_name!r} has no selective rows for {module_name!r}."
            )
        actual_ids = list(token_adapter.token_indices[adapter_name])
        if actual_ids != list(expected_ids):
            raise RuntimeError(
                f"Reloaded adapter {adapter_name!r} targets {actual_ids} for {module_name!r}, "
                f"expected {expected_ids}."
            )
        if adapter_name not in token_adapter.trainable_tokens_delta:
            raise RuntimeError(
                f"Reloaded adapter {adapter_name!r} has no selective delta for {module_name!r}."
            )
        if token_adapter.trainable_tokens_delta[adapter_name].shape[0] != len(expected_ids):
            raise RuntimeError(
                f"Reloaded adapter {adapter_name!r} has the wrong selective-row count for {module_name!r}."
            )
        wrappers[module_name] = wrapper

    if not metadata.get("weights_tied", False) or not trainable_token_indices:
        return
    input_name = metadata.get("input_embedding_module")
    output_name = metadata.get("output_head_module")
    if not input_name or not output_name:
        raise RuntimeError("Tied adapter round-trip verification requires input/output module names.")
    input_wrapper = wrappers.get(input_name) or _find_token_wrapper(named_modules, input_name)
    output_wrapper = _find_token_wrapper(named_modules, output_name)
    input_adapter = input_wrapper.token_adapter
    output_adapter = output_wrapper.token_adapter
    if output_adapter.tied_adapter is not input_adapter:
        raise RuntimeError("Reloaded tied output adapter does not point to the live input token adapter.")
    if (
        output_adapter.token_indices is not input_adapter.token_indices
        or output_adapter.trainable_tokens_original is not input_adapter.trainable_tokens_original
        or output_adapter.trainable_tokens_delta is not input_adapter.trainable_tokens_delta
    ):
        raise RuntimeError("Reloaded tied output adapter has independent selective-row state.")
    if (
        output_adapter.trainable_tokens_delta[adapter_name]
        is not input_adapter.trainable_tokens_delta[adapter_name]
    ):
        raise RuntimeError("Reloaded tied output adapter has an independent duplicate delta.")


def verify_saved_adapter_roundtrip(
    model: nn.Module,
    output_dir: Path | str,
    metadata: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Reload and compare the saved PEFT adapter without reloading the base.

    The saved adapter is attached to the already-live PEFT model under a
    temporary name. The complete adapter state returned by PEFT (LoRA and
    selective-token deltas, plus any adapter-owned saved modules) is compared
    tensor-for-tensor with the live adapter. The temporary adapter is then
    deleted and the original active adapter restored. This deliberately does
    not claim to verify a fresh base-model reload.
    """
    if metadata is None or not metadata["resolved_config"]["verify_adapter_roundtrip"]:
        return None
    required_methods = ("add_adapter", "load_adapter", "delete_adapter", "set_adapter")
    missing = [name for name in required_methods if not callable(getattr(model, name, None))]
    if missing or not isinstance(getattr(model, "peft_config", None), dict):
        raise TypeError(
            "verify_adapter_roundtrip=true requires a mutable PEFT model with add_adapter, "
            "load_adapter, delete_adapter, set_adapter, and peft_config. Missing: "
            + ", ".join(missing or ["peft_config"])
        )
    active_names = _active_adapter_names(model)
    if len(active_names) != 1:
        raise RuntimeError(
            "Adapter round-trip verification requires exactly one active source adapter; "
            f"found {active_names!r}."
        )
    source_adapter = active_names[0]
    temporary_adapter = "__special_token_roundtrip_verify__"
    if temporary_adapter in model.peft_config:
        raise RuntimeError(f"Temporary verification adapter name {temporary_adapter!r} already exists.")

    source_state = _adapter_only_live_state(model, source_adapter)
    if not source_state:
        raise RuntimeError("The live PEFT adapter state is empty; nothing can be round-trip verified.")
    artifact_state = _adapter_artifact_state(output_dir)
    _reject_full_vocab_tensors(artifact_state, metadata)
    _compare_adapter_states(source_state, artifact_state, actual_description="On-disk")
    module_topology_before = {name: id(module) for name, module in model.named_modules()}
    tied_output_adapter = None
    tied_output_base_layer = None
    if metadata.get("weights_tied", False) and metadata.get("trainable_token_indices"):
        output_name = metadata.get("output_head_module")
        if not output_name:
            raise RuntimeError("Tied adapter round-trip verification requires an output module name.")
        output_wrapper = _find_token_wrapper(dict(model.named_modules()), output_name)
        tied_output_adapter = output_wrapper.token_adapter
        tied_output_base_layer = tied_output_adapter.base_layer
    temporary_loaded = False
    try:
        # PeftModel.from_pretrained creates and autocasts its initial adapter
        # before loading checkpoint tensors, so a fresh bf16 base preserves
        # float32 adapter bytes exactly. PeftModel.load_adapter instead creates
        # a second adapter in bf16, copies into it (rounding), and only then
        # autocasts to float32. Reproduce the fresh-load ordering explicitly:
        # precreate, autocast, then ask PEFT to load into the existing adapter.
        try:
            from peft import PeftConfig
        except ImportError as exc:
            raise RuntimeError("Adapter round-trip verification requires PEFT.") from exc
        temporary_config = PeftConfig.from_pretrained(str(output_dir))
        temporary_config.inference_mode = True
        model.add_adapter(temporary_adapter, temporary_config)
        cast_adapter_dtype = getattr(
            getattr(model, "base_model", None), "_cast_adapter_dtype", None
        )
        if not callable(cast_adapter_dtype):
            raise TypeError(
                "Exact secondary-adapter verification requires PEFT base_model._cast_adapter_dtype()."
            )
        cast_adapter_dtype(
            adapter_name=temporary_adapter,
            autocast_adapter_dtype=True,
        )
        precreated_state = _adapter_only_live_state(model, temporary_adapter)
        if set(precreated_state) != set(artifact_state):
            raise RuntimeError(
                "Precreated verification adapter state keys differ from the saved artifact."
            )
        for key in sorted(artifact_state):
            if (
                precreated_state[key].shape != artifact_state[key].shape
                or precreated_state[key].dtype != artifact_state[key].dtype
            ):
                raise RuntimeError(
                    "Precreated verification adapter did not match saved tensor layout for "
                    f"{key!r}: {tuple(precreated_state[key].shape)}/{precreated_state[key].dtype} "
                    f"vs {tuple(artifact_state[key].shape)}/{artifact_state[key].dtype}."
                )
        model.load_adapter(
            str(output_dir),
            adapter_name=temporary_adapter,
            is_trainable=False,
        )
        temporary_loaded = True
        loaded_state = _adapter_only_live_state(model, temporary_adapter)
        _compare_adapter_states(source_state, loaded_state, actual_description="Reloaded")
        if metadata.get("trainable_token_indices") and not any(
            "trainable_tokens_delta" in key for key in source_state
        ):
            raise RuntimeError("Saved adapter state omitted the configured selective-token deltas.")
        _verify_loaded_adapter_rows(model, metadata, temporary_adapter)
        return {
            "method": "temporary_adapter_precreate_autocast_then_exact_tensor_compare",
            "requested": True,
            "result": "passed",
            "source_adapter": source_adapter,
            "temporary_adapter": temporary_adapter,
            "compared_tensor_count": len(source_state),
            "compared_keys": sorted(source_state),
            "parameter_preparation": "peft_add_adapter_then_cast_before_load",
        }
    finally:
        try:
            if temporary_loaded or temporary_adapter in model.peft_config:
                model.delete_adapter(temporary_adapter)
        finally:
            # PEFT currently leaves a nested TrainableTokens wrapper under a
            # tied output adapter after deleting a second adapter. Restore the
            # exact pre-verification module object rather than allowing that
            # structural mutation to leak into the live training model.
            if (
                tied_output_adapter is not None
                and tied_output_base_layer is not None
                and tied_output_adapter.base_layer is not tied_output_base_layer
            ):
                tied_output_adapter.base_layer = tied_output_base_layer
            model.set_adapter(source_adapter)
        if _active_adapter_names(model) != [source_adapter]:
            raise RuntimeError("Failed to restore the original active PEFT adapter after verification.")
        module_topology_after = {name: id(module) for name, module in model.named_modules()}
        if module_topology_after != module_topology_before:
            raise RuntimeError(
                "Adapter round-trip verification changed the live model module topology."
            )
