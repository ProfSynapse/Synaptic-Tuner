"""
Utility functions for LoRA surgery operations.

Location: shared/evolutionary/surgery/utils.py
Purpose: Weight I/O, adapter management, key parsing, and math helpers
         shared across all surgery operations.
Used by: All operation classes in surgery/operations/, LoRASurgeon, tests.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from typing import Any, Dict, List, Tuple

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from safetensors.torch import load_file as st_load_file, save_file as st_save_file
except ImportError:
    st_load_file = None  # type: ignore[assignment]
    st_save_file = None  # type: ignore[assignment]


def check_dependencies() -> None:
    """Verify that required dependencies are available."""
    if torch is None:
        raise ImportError(
            "PyTorch is required for LoRA surgery. Install with: pip install torch"
        )
    if st_load_file is None or st_save_file is None:
        raise ImportError(
            "safetensors is required for LoRA surgery. Install with: pip install safetensors"
        )


def load_adapter_config(adapter_dir: str) -> Dict[str, Any]:
    """Load adapter_config.json from an adapter directory."""
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")
    with open(config_path, "r") as fh:
        return json.load(fh)


def save_adapter_config(adapter_dir: str, config: Dict[str, Any]) -> None:
    """Save adapter_config.json to an adapter directory."""
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2)


def find_safetensor_files(adapter_dir: str) -> List[str]:
    """Find all safetensor weight files in an adapter directory."""
    files = []
    for fname in os.listdir(adapter_dir):
        if fname.endswith(".safetensors"):
            files.append(os.path.join(adapter_dir, fname))
    return sorted(files)


def load_all_weights(adapter_dir: str) -> Dict[str, "torch.Tensor"]:
    """Load all weights from safetensor files in an adapter directory."""
    check_dependencies()
    all_weights: Dict[str, torch.Tensor] = {}
    for fpath in find_safetensor_files(adapter_dir):
        all_weights.update(st_load_file(fpath))
    return all_weights


def save_all_weights(adapter_dir: str, weights: Dict[str, "torch.Tensor"]) -> None:
    """Save weights to a single safetensor file in the adapter directory."""
    check_dependencies()
    out_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    st_save_file(weights, out_path)


def copy_adapter(src: str, dst: str) -> str:
    """Copy an adapter directory, returning the destination path."""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def get_layer_indices(weight_keys: List[str]) -> List[int]:
    """Extract unique layer indices from weight key names.

    Typical key pattern: ``base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight``
    """
    indices = set()
    pattern = re.compile(r"layers\.(\d+)\.")
    for key in weight_keys:
        match = pattern.search(key)
        if match:
            indices.add(int(match.group(1)))
    return sorted(indices)


def get_module_types(weight_keys: List[str]) -> List[str]:
    """Extract unique module type names (q_proj, k_proj, ...) from weight keys."""
    types = set()
    pattern = re.compile(r"\.(\w+_proj)\.lora_[AB]")
    for key in weight_keys:
        match = pattern.search(key)
        if match:
            types.add(match.group(1))
    return sorted(types)


def is_attention_key(key: str) -> bool:
    """Return True if key belongs to an attention module."""
    attention_modules = {"q_proj", "k_proj", "v_proj", "o_proj"}
    for mod in attention_modules:
        if f".{mod}." in key:
            return True
    return False


def is_mlp_key(key: str) -> bool:
    """Return True if key belongs to an MLP module."""
    mlp_modules = {"gate_proj", "up_proj", "down_proj"}
    for mod in mlp_modules:
        if f".{mod}." in key:
            return True
    return False


def softmax(values: List[float], temperature: float = 1.0) -> List[float]:
    """Compute softmax over a list of floats with temperature scaling."""
    scaled = [v / temperature for v in values]
    max_val = max(scaled)
    exps = [math.exp(v - max_val) for v in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def find_lora_pairs(
    weights: Dict[str, "torch.Tensor"],
) -> Dict[str, Tuple[str, str]]:
    """Find matching LoRA A/B weight pairs.

    Returns:
        Dict mapping a common prefix to (a_key, b_key) tuples.
    """
    a_keys = {}
    b_keys = {}
    for key in weights:
        if ".lora_A." in key:
            prefix = key.split(".lora_A.")[0]
            a_keys[prefix] = key
        elif ".lora_B." in key:
            prefix = key.split(".lora_B.")[0]
            b_keys[prefix] = key

    pairs = {}
    for prefix in a_keys:
        if prefix in b_keys:
            pairs[prefix] = (a_keys[prefix], b_keys[prefix])

    return pairs


