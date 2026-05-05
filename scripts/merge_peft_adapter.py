#!/usr/bin/env python3
"""Merge a PEFT adapter into a base model and save a merged HF model."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES = {
    "auto": None,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="Base merged model path or HF model id")
    parser.add_argument("--adapter", required=True, type=Path, help="PEFT adapter path")
    parser.add_argument("--output", required=True, type=Path, help="Merged model output path")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-shard-size", default="2GB")
    parser.add_argument(
        "--align-saved-keys-to-base",
        action="store_true",
        help=(
            "After saving, normalize saved checkpoint key paths only when the "
            "repaired key exists in the base checkpoint. Useful for adapter "
            "auto-mapping loaders that save nested module prefixes."
        ),
    )
    parser.add_argument(
        "--copy-missing-base-keys",
        action="store_true",
        help=(
            "After optional key alignment, copy any base checkpoint tensors "
            "that are still missing from the merged output into an additional "
            "safetensors shard."
        ),
    )
    parser.add_argument(
        "--base-loader",
        choices=["auto-causal", "adapter-auto-mapping"],
        default="auto-causal",
        help="How to load the base model before applying the adapter.",
    )
    parser.add_argument("--no-trust-remote-code", action="store_true")
    return parser.parse_args()


def load_adapter_mapped_base(base: Path, adapter: Path, dtype, device_map: str):
    config_path = adapter / "adapter_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        adapter_config = json.load(f)
    auto_mapping = adapter_config.get("auto_mapping") or {}
    parent_library = auto_mapping.get("parent_library")
    class_name = auto_mapping.get("base_model_class")
    if not parent_library or not class_name:
        raise ValueError(f"Adapter config does not define auto_mapping: {config_path}")

    module = importlib.import_module(parent_library)
    model_class = getattr(module, class_name)
    return model_class.from_pretrained(
        str(base),
        torch_dtype=dtype,
        device_map=device_map,
    )


def _iter_safetensor_paths(model_dir: Path) -> list[Path]:
    return sorted(model_dir.glob("*.safetensors"))


def _read_safetensor_keys(model_dir: Path) -> set[str]:
    keys: set[str] = set()
    for path in _iter_safetensor_paths(model_dir):
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys.update(handle.keys())
    return keys


def _collapse_adjacent_duplicate_segments(key: str) -> str:
    parts = key.split(".")
    collapsed: list[str] = []
    for part in parts:
        if collapsed and collapsed[-1] == part:
            continue
        collapsed.append(part)
    return ".".join(collapsed)


def _base_aligned_candidate(key: str, base_keys: set[str]) -> str | None:
    if key in base_keys:
        return key

    collapsed = _collapse_adjacent_duplicate_segments(key)
    if collapsed in base_keys:
        return collapsed

    parts = collapsed.split(".")
    candidates: set[str] = set()
    for index in range(len(parts)):
        candidate = ".".join(parts[:index] + parts[index + 1 :])
        if candidate in base_keys:
            candidates.add(candidate)
    if len(candidates) == 1:
        return next(iter(candidates))
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous base-aligned key candidates for {key}: {sorted(candidates)}")
    return None


def align_saved_keys_to_base(output: Path, base: Path) -> int:
    base_keys = _read_safetensor_keys(base)
    if not base_keys:
        raise ValueError(f"No safetensor keys found in base model: {base}")

    rename_map: dict[str, str] = {}
    for path in _iter_safetensor_paths(output):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in base_keys:
                    continue
                candidate = _base_aligned_candidate(key, base_keys)
                if candidate and candidate != key:
                    rename_map[key] = candidate

    if not rename_map:
        return 0

    renamed_values = set(rename_map.values())
    if len(renamed_values) != len(rename_map):
        raise ValueError("Key normalization would create duplicate target keys")

    for path in _iter_safetensor_paths(output):
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            mapped_keys = [rename_map.get(key, key) for key in handle.keys()]
            if len(mapped_keys) != len(set(mapped_keys)):
                raise ValueError(f"Key normalization collision in {path}")
            tensors = {
                rename_map.get(key, key): handle.get_tensor(key)
                for key in handle.keys()
            }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        save_file(tensors, tmp_path, metadata=metadata)
        tmp_path.replace(path)

    index_path = output / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map")
        if isinstance(weight_map, dict):
            index["weight_map"] = {
                rename_map.get(key, key): value
                for key, value in weight_map.items()
            }
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2, sort_keys=True)

    return len(rename_map)


def copy_missing_base_keys(output: Path, base: Path) -> int:
    base_keys = _read_safetensor_keys(base)
    output_keys = _read_safetensor_keys(output)
    missing_keys = base_keys - output_keys
    if not missing_keys:
        return 0

    missing_tensors: dict[str, torch.Tensor] = {}
    for path in _iter_safetensor_paths(base):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in missing_keys:
                    missing_tensors[key] = handle.get_tensor(key)

    if set(missing_tensors) != missing_keys:
        unresolved = sorted(missing_keys - set(missing_tensors))
        raise ValueError(f"Could not find missing base tensors: {unresolved[:10]}")

    shard_name = "model-base-missing.safetensors"
    save_file(missing_tensors, output / shard_name)

    index_path = output / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.setdefault("weight_map", {})
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid weight_map in {index_path}")
        for key in sorted(missing_tensors):
            weight_map[key] = shard_name
        metadata = index.setdefault("metadata", {})
        if isinstance(metadata, dict) and "total_size" in metadata:
            metadata["total_size"] = int(metadata["total_size"]) + sum(
                tensor.numel() * tensor.element_size()
                for tensor in missing_tensors.values()
            )
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, sort_keys=True)

    return len(missing_tensors)


def main() -> int:
    args = parse_args()
    trust_remote_code = not args.no_trust_remote_code
    dtype = DTYPES[args.dtype]

    print(f"Loading base: {args.base}")
    if args.base_loader == "adapter-auto-mapping":
        model = load_adapter_mapped_base(args.base, args.adapter, dtype, args.device_map)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(args.base),
            torch_dtype=dtype,
            device_map=args.device_map,
            trust_remote_code=trust_remote_code,
        )

    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, str(args.adapter))

    print("Merging adapter")
    model = model.merge_and_unload()

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model: {args.output}")
    model.save_pretrained(
        str(args.output),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    if args.align_saved_keys_to_base:
        renamed_count = align_saved_keys_to_base(args.output, args.base)
        print(f"Aligned saved checkpoint keys to base model: {renamed_count} renamed")
    if args.copy_missing_base_keys:
        copied_count = copy_missing_base_keys(args.output, args.base)
        print(f"Copied missing base checkpoint keys: {copied_count}")

    tokenizer_source = args.adapter if (args.adapter / "tokenizer_config.json").exists() else args.base
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source),
        trust_remote_code=trust_remote_code,
    )
    tokenizer.save_pretrained(str(args.output))

    for filename in ("processor_config.json", "preprocessor_config.json"):
        source = args.adapter / filename
        if not source.exists():
            source = args.base / filename
        if source.exists():
            shutil.copy2(source, args.output / filename)

    print(f"Merged model saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
