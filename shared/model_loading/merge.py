"""
Centralized LoRA merge utilities.

Location: shared/model_loading/merge.py
Purpose: Merge LoRA adapters into base model weights
Used by: GRPO trainer, merge_handler, upload workflows

This provides composable merge functionality:
  1. Train SFT/KTO → LoRA checkpoint
  2. Merge → merged-16bit (this module)
  3. Train GRPO → new LoRA on merged base
  4. Upload
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, Any

import torch


def is_lora_checkpoint(path: Path) -> bool:
    """Check if path is a LoRA checkpoint (has adapter_config.json)."""
    return (path / "adapter_config.json").exists()


def is_merged_model(path: Path) -> bool:
    """Check if path is a merged model (has config.json, no adapter_config)."""
    return (path / "config.json").exists() and not is_lora_checkpoint(path)


def get_base_model_name(lora_path: Path) -> str:
    """Extract base model name from LoRA adapter config."""
    adapter_config = lora_path / "adapter_config.json"
    if adapter_config.exists():
        with open(adapter_config) as f:
            config = json.load(f)
            base_model = config.get("base_model_name_or_path", "")
            if "/" in base_model:
                return base_model.split("/")[-1]
    return "merged-model"


def find_merged_for_run(run_path: Path) -> Optional[Path]:
    """
    Find existing merged model in a training run directory.

    Looks for model-name/merged-16bit pattern.

    Args:
        run_path: Path to training run directory

    Returns:
        Path to merged model if found, None otherwise
    """
    # Look for model-name/merged-16bit pattern
    for subdir in run_path.iterdir():
        if subdir.is_dir() and subdir.name not in ('checkpoints', 'logs', 'final_model'):
            merged = subdir / "merged-16bit"
            if merged.exists() and (merged / "config.json").exists():
                return merged

    # Also check direct merged-16bit in run
    merged = run_path / "merged-16bit"
    if merged.exists() and (merged / "config.json").exists():
        return merged

    return None


# Family literals for the merge seam. "causal_lm" is the historical default and
# its dispatch path is held BEHAVIOR-IDENTICAL to the pre-seam code. "embedding"
# dispatches to the sentence-transformers / FastSentenceTransformer merge path.
# These literals match the canonical method/family strings used across the repo
# (no variants). merge.py does NOT read the registry — the family is passed in by
# the caller that knows it (keeps shared/ import-light and free of Trainers/ deps).
MERGE_FAMILY_CAUSAL_LM = "causal_lm"
MERGE_FAMILY_EMBEDDING = "embedding"
SUPPORTED_MERGE_FAMILIES = (MERGE_FAMILY_CAUSAL_LM, MERGE_FAMILY_EMBEDDING)


def _merge_causal_lm(
    lora_path: Path,
    output_path: Path,
    max_seq_length: int,
    load_in_4bit: bool,
) -> None:
    """Causal-LM merge path (DEFAULT). Held behavior-identical to the pre-seam code.

    Loads the LoRA checkpoint with unsloth.FastLanguageModel and saves a merged
    16-bit model. SFT/KTO/GRPO merges flow through here unchanged.
    """
    from unsloth import FastLanguageModel

    print("\nLoading LoRA checkpoint...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(lora_path),
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    print("Saving merged 16-bit model...")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(output_path), tokenizer, save_method="merged_16bit")

    # Clear GPU memory
    del model
    del tokenizer
    torch.cuda.empty_cache()


def _merge_embedding(
    lora_path: Path,
    output_path: Path,
    max_seq_length: int,
) -> None:
    """Embedding merge path. Mirrors the causal-LM structure via the ST fast path.

    Loads the LoRA checkpoint with unsloth.FastSentenceTransformer (which exposes
    save_pretrained_merged, just like FastLanguageModel) and saves a merged
    16-bit sentence-transformers model. `load_in_4bit` is intentionally NOT
    accepted here: QLoRA is deferred for embedding in v1 (R8), and the ST loader
    does not take a 4-bit flag — so the dispatch must never pass one through.
    """
    from unsloth import FastSentenceTransformer

    print("\nLoading embedding LoRA checkpoint (sentence-transformers)...")
    model = FastSentenceTransformer.from_pretrained(
        str(lora_path),
        max_seq_length=max_seq_length,
    )

    print("Saving merged 16-bit embedding model...")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(output_path), save_method="merged_16bit")

    # Clear GPU memory
    del model
    torch.cuda.empty_cache()


def _merge_loader_for_family(family: str):
    """Return the family-specific merge implementation.

    "causal_lm" (default) -> _merge_causal_lm (unsloth FastLanguageModel path)
    "embedding"           -> _merge_embedding (ST/FastSentenceTransformer path)
    Unknown family        -> ValueError.
    """
    dispatch = {
        MERGE_FAMILY_CAUSAL_LM: _merge_causal_lm,
        MERGE_FAMILY_EMBEDDING: _merge_embedding,
    }
    if family not in dispatch:
        raise ValueError(
            f"Unknown merge family {family!r}; must be one of {list(SUPPORTED_MERGE_FAMILIES)}"
        )
    return dispatch[family]


def merge_lora_checkpoint(
    lora_path: Path,
    output_path: Path,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    family: str = MERGE_FAMILY_CAUSAL_LM,   # NEW PARAM, default preserves existing behavior
) -> Path:
    """
    Merge LoRA adapters into base model.

    Args:
        lora_path: Path to LoRA checkpoint
        output_path: Path to save merged model
        max_seq_length: Maximum sequence length
        load_in_4bit: Whether to load in 4-bit (causal_lm only; ignored for embedding)
        family: Model family — "causal_lm" (default, behavior-identical to the
            pre-seam code) or "embedding". The caller that knows the family
            supplies it; merge.py does not read the registry.

    Returns:
        Path to merged model
    """
    print("\n" + "=" * 60)
    print("MERGING LORA CHECKPOINT")
    print("=" * 60)
    print(f"LoRA path: {lora_path}")
    print(f"Output: {output_path}")
    print(f"Family: {family}")

    merge_fn = _merge_loader_for_family(family)
    if family == MERGE_FAMILY_EMBEDDING:
        # load_in_4bit is ignored for embedding (QLoRA deferred, R8) — do not
        # thread a 4-bit flag into the ST loader that does not support one.
        merge_fn(lora_path, output_path, max_seq_length)
    else:
        merge_fn(lora_path, output_path, max_seq_length, load_in_4bit)

    print(f"✓ Merged model saved to: {output_path}")
    print("=" * 60 + "\n")

    return output_path


def find_or_create_merged(lora_path: Path, max_seq_length: int = 2048) -> Path:
    """
    Find existing merged model or create one from LoRA checkpoint.

    This is the main entry point for auto-merge functionality.
    Checks if a merged model already exists for this run, and if not,
    creates one from the LoRA checkpoint.

    Args:
        lora_path: Path to LoRA checkpoint
        max_seq_length: Maximum sequence length for loading

    Returns:
        Path to merged model (existing or newly created)
    """
    # Determine run directory
    run_dir = lora_path.parent
    if run_dir.name in ('checkpoints', 'final_model'):
        run_dir = run_dir.parent

    # Look for existing merged model
    existing = find_merged_for_run(run_dir)
    if existing:
        print(f"Found existing merged model: {existing}")
        return existing

    # No merged model found - create one
    model_name = get_base_model_name(lora_path)
    output_dir = run_dir / model_name / "merged-16bit"

    return merge_lora_checkpoint(
        lora_path=lora_path,
        output_path=output_dir,
        max_seq_length=max_seq_length,
    )


def resolve_model_path(
    model_path: str,
    max_seq_length: int = 2048,
) -> Tuple[Path, bool]:
    """
    Resolve a model path, auto-merging if it's a LoRA checkpoint.

    This handles the common pattern where a user provides either:
    - A LoRA checkpoint path → auto-merge and return merged path
    - A merged model path → return as-is

    Args:
        model_path: Path to model (LoRA checkpoint or merged model)
        max_seq_length: Maximum sequence length for merge operation

    Returns:
        Tuple of (resolved_path, was_merged)
        - resolved_path: Path to merged model
        - was_merged: True if merge was performed

    Raises:
        FileNotFoundError: If model_path doesn't exist
        ValueError: If path is neither LoRA checkpoint nor merged model
    """
    # Always use absolute paths (PEFT saves base_model_name_or_path as-is)
    path = Path(model_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Check if LoRA checkpoint
    if is_lora_checkpoint(path):
        print(f"Detected LoRA checkpoint at: {model_path}")
        merged_path = find_or_create_merged(path, max_seq_length)
        return merged_path, True

    # Check if merged model
    if is_merged_model(path):
        return path, False

    raise ValueError(
        f"Path '{model_path}' doesn't appear to be a valid model.\n"
        f"Expected a LoRA checkpoint (with adapter_config.json) or "
        f"merged model (with config.json)"
    )
