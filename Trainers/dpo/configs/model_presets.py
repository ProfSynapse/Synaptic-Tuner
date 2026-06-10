"""
Model preset map for the DPO trainer.

DEVIATION FROM KTO (flagged): Trainers/kto/train_kto.py defines its model_map
inline inside main(), so preset resolution cannot be exercised without importing
the full trl/unsloth/torch stack. This module lifts the equivalent map to an
importable, dependency-free location so a config-validation dry-run (and its
smoke test) can resolve a friendly flag (e.g. --qwen3-4b) to a (size, repo)
pair without loading any ML library. train_dpo.py imports MODEL_MAP /
resolve_model_flag from here.

Each entry maps a friendly argparse flag name (underscored, matching the
store_true dest) to a (model_size, hf_repo) tuple. Qwen3-4B/8B are the Phase 1
pins (Apache-2.0, ungated). The repo names use the unsloth bnb-4bit mirrors,
matching the convention of every other entry in the map; CODE/operators should
confirm the exact unsloth/Qwen3-* repo names resolve on live HF before a real
run (model-landscape doc confirms the upstream Qwen3-4B/8B-Instruct exist).
"""

from typing import Dict, Optional, Tuple

# flag_dest -> (model_size, hf_repo)
MODEL_MAP: Dict[str, Tuple[str, str]] = {
    # 3-4B models (fast iteration)
    'qwen_3b': ('3b', 'unsloth/Qwen2.5-3B-Instruct-bnb-4bit'),
    'llama_3b': ('3b', 'unsloth/Llama-3.2-3B-Instruct-bnb-4bit'),
    'qwen3_4b': ('3b', 'unsloth/Qwen3-4B-Instruct-bnb-4bit'),

    # 7-8B models (production quality)
    'mistral_7b': ('7b', 'unsloth/mistral-7b-v0.3-bnb-4bit'),
    'llama_8b': ('7b', 'unsloth/llama-3.1-8b-instruct-bnb-4bit'),
    'qwen_7b': ('7b', 'unsloth/Qwen2.5-7B-Instruct-bnb-4bit'),
    'qwen3_8b': ('7b', 'unsloth/Qwen3-8B-Instruct-bnb-4bit'),

    # 11-14B models (advanced)
    'llama_13b': ('13b', 'unsloth/llama-2-13b-bnb-4bit'),
    'gemma_12b': ('13b', 'unsloth/gemma-3-12b-it-unsloth-bnb-4bit'),

    # 17-24B models (very large)
    'gpt_20b': ('20b', 'unsloth/gpt-oss-20b-unsloth-bnb-4bit'),
    'mistral_24b': ('20b', 'unsloth/Mistral-Small-3.2-24B-Instruct-2506-unsloth-bnb-4bit'),
}


def resolve_model_flag(args) -> Optional[Tuple[str, str]]:
    """Return (model_size, hf_repo) for the first set friendly flag, else None.

    Args:
        args: argparse.Namespace whose attributes match MODEL_MAP keys.

    Returns:
        (model_size, hf_repo) for the first truthy flag in MODEL_MAP order,
        or None if no friendly flag is set (caller falls back to config / --model-name).
    """
    for flag, (size, model_name) in MODEL_MAP.items():
        if getattr(args, flag, False):
            return size, model_name
    return None
