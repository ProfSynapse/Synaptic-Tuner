#!/usr/bin/env python3
"""Fallback local LoRA merge for adapters PEFT cannot attach to cleanly.

This CLI applies LoRA deltas directly to matching model modules. It is intended
for local Mac workflows where:
- the adapter weights are valid
- the target layers are real linear weights in the base model
- PEFT fails because the broader model tree includes unsupported wrapper types

It is not a replacement for the normal PEFT/Unsloth merge path. Use this only
when the adapter keys and target modules are well understood.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

# When executed as a script from shared/upload/, sys.path[0] points at a
# directory containing a local `platform` package, which can shadow the Python
# stdlib `platform` module during torch import. Remove the script directory from
# the import path and ensure the repo root is present for package imports.
script_dir_str = str(SCRIPT_DIR)
repo_root_str = str(REPO_ROOT)
if script_dir_str in sys.path:
    sys.path.remove(script_dir_str)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

import torch
from safetensors.torch import save_file as save_safetensors
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from shared.evolutionary.surgery.utils import (
    find_lora_pairs,
    load_adapter_config,
    load_all_weights,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="HF model id or local base model path.")
    parser.add_argument("--adapter-dir", required=True, help="Adapter directory containing adapter_config.json.")
    parser.add_argument("--output-dir", required=True, help="Where to save merged model/tokenizer.")
    parser.add_argument(
        "--loader",
        choices=("causal", "conditional"),
        default="causal",
        help="Model loader family to use for the base model.",
    )
    parser.add_argument(
        "--include-substring",
        action="append",
        default=[],
        help="Only merge adapter keys containing this substring. Repeatable.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Transformers loaders.",
    )
    parser.add_argument(
        "--save-state-prefix",
        action="append",
        default=[],
        help="Save only state_dict keys containing one of these prefixes instead of calling model.save_pretrained. Repeatable.",
    )
    return parser


def load_model(base_model: str, loader: str, trust_remote_code: bool):
    common = {
        "pretrained_model_name_or_path": base_model,
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
    }
    if loader == "conditional":
        return AutoModelForImageTextToText.from_pretrained(**common)
    return AutoModelForCausalLM.from_pretrained(**common)


def candidate_module_paths(adapter_prefix: str) -> Iterable[str]:
    # Typical PEFT key:
    # base_model.model.model.language_model.layers.0.self_attn.q_proj
    candidates = [adapter_prefix]
    prefixes = [
        "base_model.model.",
        "base_model.",
        "model.",
    ]
    for prefix in prefixes:
        if adapter_prefix.startswith(prefix):
            candidates.append(adapter_prefix[len(prefix):])
    # Special common case for PEFT-wrapped multimodal models.
    if adapter_prefix.startswith("base_model.model.model."):
        candidates.append("model." + adapter_prefix[len("base_model.model.model."):])
    # Deduplicate while preserving order.
    seen = set()
    for item in candidates:
        if item not in seen:
            seen.add(item)
            yield item


def resolve_target_module(model, adapter_prefix: str):
    modules = dict(model.named_modules())
    for candidate in candidate_module_paths(adapter_prefix):
        module = modules.get(candidate)
        if module is not None:
            return candidate, module
    return None, None


def get_target_weight(module: torch.nn.Module) -> torch.nn.Parameter:
    if hasattr(module, "weight") and isinstance(module.weight, torch.nn.Parameter):
        return module.weight
    if hasattr(module, "linear") and hasattr(module.linear, "weight") and isinstance(module.linear.weight, torch.nn.Parameter):
        return module.linear.weight
    raise TypeError(f"Module {type(module).__name__} does not expose a mergeable weight")


def merge_lora_into_model(
    *,
    model,
    adapter_dir: Path,
    include_substrings: list[str],
) -> tuple[int, list[str]]:
    weights = load_all_weights(str(adapter_dir))
    config = load_adapter_config(str(adapter_dir))
    scaling = config["lora_alpha"] / config["r"]
    pairs = find_lora_pairs(weights)

    merged = 0
    skipped: list[str] = []

    with torch.no_grad():
        for prefix, (a_key, b_key) in pairs.items():
            if include_substrings and not any(fragment in prefix for fragment in include_substrings):
                skipped.append(prefix)
                continue

            _, module = resolve_target_module(model, prefix)
            if module is None:
                skipped.append(prefix)
                continue

            target_weight = get_target_weight(module)
            a = weights[a_key].to(dtype=torch.float32)
            b = weights[b_key].to(dtype=torch.float32)
            delta = (b @ a) * scaling
            delta = delta.to(dtype=target_weight.dtype, device=target_weight.device)

            if target_weight.shape != delta.shape:
                raise ValueError(
                    f"Delta shape mismatch for {prefix}: weight={tuple(target_weight.shape)} delta={tuple(delta.shape)}"
                )

            target_weight.add_(delta)
            merged += 1

    return merged, skipped


def save_model_artifacts(
    *,
    model,
    tokenizer,
    output_dir: Path,
    save_state_prefixes: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("saving config...")
    model.config.save_pretrained(str(output_dir))
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(str(output_dir))

    print("saving tokenizer...")
    tokenizer.save_pretrained(str(output_dir))

    if save_state_prefixes:
        print("saving filtered state dict...")
        state_dict = model.state_dict()
        filtered = {
            key: tensor.detach().cpu().contiguous()
            for key, tensor in state_dict.items()
            if any(prefix in key for prefix in save_state_prefixes)
        }
        if not filtered:
            raise RuntimeError(
                f"No state_dict keys matched save-state-prefixes: {save_state_prefixes}"
            )
        save_safetensors(filtered, str(output_dir / "model.safetensors"))
        print(f"saved filtered tensors: {len(filtered)}")
        return

    print("saving merged model...")
    model.save_pretrained(str(output_dir), safe_serialization=True)


def main() -> int:
    args = build_parser().parse_args()
    adapter_dir = Path(args.adapter_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    print("loading base model...")
    model = load_model(args.base_model, args.loader, args.trust_remote_code)
    print(type(model).__name__)

    print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)

    print("merging LoRA deltas directly...")
    merged_count, skipped = merge_lora_into_model(
        model=model,
        adapter_dir=adapter_dir,
        include_substrings=list(args.include_substring),
    )
    print(f"merged modules: {merged_count}")
    if skipped:
        print(f"skipped prefixes: {len(skipped)}")
        for item in skipped[:10]:
            print(f"  skipped: {item}")

    save_model_artifacts(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        save_state_prefixes=list(args.save_state_prefix),
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
